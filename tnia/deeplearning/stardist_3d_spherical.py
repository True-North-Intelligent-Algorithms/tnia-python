from stardist.models import StarDist3D
import functools
from stardist.rays3d import rays_from_json
from stardist.nms import non_maximum_suppression_3d, non_maximum_suppression_3d_sparse
from stardist.geometry import polyhedron_to_label
from stardist.matching import relabel_sequential
import numpy as np
# import rastor
import raster_geometry as rg
import math

def polyhedron_to_sphere_labels(points, shape):
    labels = np.zeros(shape, dtype=np.float32)
    r = 50
    size = [2*r, 2*r, 2*r]
    label_num=1
    for point in points:    
        sphere = rg.sphere(size, r).astype(np.float32)
        add_small_to_large(labels, label_num*sphere, point[2], point[1], point[0])
        label_num += 1

    return labels.astype(np.int32)

class StarDist3DSpherical(StarDist3D):

    def _predict_instances_generator(self, img, axes=None, normalizer=None,
                                        sparse=True,
                                        prob_thresh=None, nms_thresh=None,
                                        scale=None,
                                        n_tiles=None, show_tile_progress=True,
                                        verbose=False,
                                        return_labels=True,
                                        predict_kwargs=None, nms_kwargs=None,
                                        overlap_label=None, return_predict=False):
            """Predict instance segmentation from input image.

            Parameters
            ----------
            img : :class:`numpy.ndarray`
                Input image
            axes : str or None
                Axes of the input ``img``.
                ``None`` denotes that axes of img are the same as denoted in the config.
            normalizer : :class:`csbdeep.data.Normalizer` or None
                (Optional) normalization of input image before prediction.
                Note that the default (``None``) assumes ``img`` to be already normalized.
            sparse: bool
                If true, aggregate probabilities/distances sparsely during tiled
                prediction to save memory (recommended).
            prob_thresh : float or None
                Consider only object candidates from pixels with predicted object probability
                above this threshold (also see `optimize_thresholds`).
            nms_thresh : float or None
                Perform non-maximum suppression that considers two objects to be the same
                when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
            scale: None or float or iterable
                Scale the input image internally by this factor and rescale the output accordingly.
                All spatial axes (X,Y,Z) will be scaled if a scalar value is provided.
                Alternatively, multiple scale values (compatible with input `axes`) can be used
                for more fine-grained control (scale values for non-spatial axes must be 1).
            n_tiles : iterable or None
                Out of memory (OOM) errors can occur if the input image is too large.
                To avoid this problem, the input image is broken up into (overlapping) tiles
                that are processed independently and re-assembled.
                This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
                ``None`` denotes that no tiling should be used.
            show_tile_progress: bool
                Whether to show progress during tiled prediction.
            verbose: bool
                Whether to print some info messages.
            return_labels: bool
                Whether to create a label image, otherwise return None in its place.
            predict_kwargs: dict
                Keyword arguments for ``predict`` function of Keras model.
            nms_kwargs: dict
                Keyword arguments for non-maximum suppression.
            overlap_label: scalar or None
                if not None, label the regions where polygons overlap with that value
            return_predict: bool
                Also return the outputs of :func:`predict` (in a separate tuple)
                If True, implies sparse = False

            Returns
            -------
            (:class:`numpy.ndarray`, dict), (optional: return tuple of :func:`predict`)
                Returns a tuple of the label instances image and also
                a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.

            """
            if predict_kwargs is None:
                predict_kwargs = {}
            if nms_kwargs is None:
                nms_kwargs = {}

            if return_predict and sparse:
                sparse = False
                warnings.warn("Setting sparse to False because return_predict is True")

            nms_kwargs.setdefault("verbose", verbose)

            _axes         = self._normalize_axes(img, axes)
            _axes_net     = self.config.axes
            _permute_axes = self._make_permute_axes(_axes, _axes_net)
            _shape_inst   = tuple(s for s,a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')

            if scale is not None:
                if isinstance(scale, numbers.Number):
                    scale = tuple(scale if a in 'XYZ' else 1 for a in _axes)
                scale = tuple(scale)
                len(scale) == len(_axes) or _raise(ValueError(f"scale {scale} must be of length {len(_axes)}, i.e. one value for each of the axes {_axes}"))
                for s,a in zip(scale,_axes):
                    s > 0 or _raise(ValueError("scale values must be greater than 0"))
                    (s in (1,None) or a in 'XYZ') or warnings.warn(f"replacing scale value {s} for non-spatial axis {a} with 1")
                scale = tuple(s if a in 'XYZ' else 1 for s,a in zip(scale,_axes))
                verbose and print(f"scaling image by factors {scale} for axes {_axes}")
                img = ndi.zoom(img, scale, order=1)

            yield 'predict'  # indicate that prediction is starting
            res = None
            if sparse:
                for res in self._predict_sparse_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                                        prob_thresh=prob_thresh, show_tile_progress=show_tile_progress, **predict_kwargs):
                    if res is None:
                        yield 'tile'  # yield 'tile' each time a tile has been processed
            else:
                for res in self._predict_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                                show_tile_progress=show_tile_progress, **predict_kwargs):
                    if res is None:
                        yield 'tile'  # yield 'tile' each time a tile has been processed
                res = tuple(res) + (None,)

            if self._is_multiclass():
                prob, dist, prob_class, points = res
            else:
                prob, dist, points = res
                prob_class = None

            yield 'nms'  # indicate that non-maximum suppression is starting
            res_instances = self._instances_from_prediction(_shape_inst, prob, dist,
                                                            points=points,
                                                            prob_class=prob_class,
                                                            prob_thresh=prob_thresh,
                                                            nms_thresh=nms_thresh,
                                                            scale=(None if scale is None else dict(zip(_axes,scale))),
                                                            return_labels=return_labels,
                                                            overlap_label=overlap_label,
                                                            **nms_kwargs)

            # last "yield" is the actual output that would have been "return"ed if this was a regular function
            if return_predict:
                yield res_instances, tuple(res[:-1])
            else:
                yield res_instances


    @functools.wraps(_predict_instances_generator)
    def predict_instances_test(self, *args, **kwargs):
        # the reason why the actual computation happens as a generator function
        # (in '_predict_instances_generator') is that the generator is called
        # from the stardist napari plugin, which has its benefits regarding
        # control flow and progress display. however, typical use cases should
        # almost always use this function ('predict_instances'), and shouldn't
        # even notice (thanks to @functools.wraps) that it wraps the generator
        # function. note that similar reasoning applies to 'predict' and
        # 'predict_sparse'.

        # return last "yield"ed value of generator
        r = None
        for r in self._predict_instances_generator(*args, **kwargs):
            pass
        return r
    
    def _instances_from_prediction(self, img_shape, prob, dist, points=None, prob_class=None, prob_thresh=None, nms_thresh=None, overlap_label=None, return_labels=True, scale=None, **nms_kwargs):
        """
        if points is None     -> dense prediction
        if points is not None -> sparse prediction

        if prob_class is None     -> single class prediction
        if prob_class is not None -> multi  class prediction
        """
        if prob_thresh is None: prob_thresh = self.thresholds.prob
        if nms_thresh  is None: nms_thresh  = self.thresholds.nms

        rays = rays_from_json(self.config.rays_json)

        # sparse prediction
        if points is not None:
            points, probi, disti, indsi = non_maximum_suppression_3d_sparse(dist, prob, points, rays, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                prob_class = prob_class[indsi]

        # dense prediction
        else:
            points, probi, disti = non_maximum_suppression_3d(dist, prob, rays, grid=self.config.grid,
                                                              prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                inds = tuple(p//g for p,g in zip(points.T, self.config.grid))
                prob_class = prob_class[inds]

        verbose = nms_kwargs.get('verbose',False)
        verbose and print("render polygons...")

        if scale is not None:
            # need to undo the scaling given by the scale dict, e.g. scale = dict(X=0.5,Y=0.5,Z=1.0):
            #   1. re-scale points (origins of polyhedra)
            #   2. re-scale vectors of rays object (computed from distances)
            if not (isinstance(scale,dict) and 'X' in scale and 'Y' in scale and 'Z' in scale):
                raise ValueError("scale must be a dictionary with entries for 'X', 'Y', and 'Z'")
            rescale = (1/scale['Z'],1/scale['Y'],1/scale['X'])
            points = points * np.array(rescale).reshape(1,3)
            rays = rays.copy(scale=rescale)
        else:
            rescale = (1,1,1)

        if return_labels:
            #labels = polyhedron_to_label(disti, points, rays=rays, prob=probi, shape=img_shape, overlap_label=overlap_label, verbose=verbose)
            labels = polyhedron_to_sphere_labels(points, shape=img_shape)

            # map the overlap_label to something positive and back
            # (as relabel_sequential doesn't like negative values)
            if overlap_label is not None and overlap_label<0 and (overlap_label in labels):
                overlap_mask = (labels == overlap_label)
                overlap_label2 = max(set(np.unique(labels))-{overlap_label})+1
                labels[overlap_mask] = overlap_label2
                labels, fwd, bwd = relabel_sequential(labels)
                labels[labels == fwd[overlap_label2]] = overlap_label
            else:
                # TODO relabel_sequential necessary?
                # print(np.unique(labels))
                labels, _,_ = relabel_sequential(labels)
                # print(np.unique(labels))
        else:
            labels = None

        res_dict = dict(dist=disti, points=points, prob=probi, rays=rays, rays_vertices=rays.vertices, rays_faces=rays.faces)

        if prob_class is not None:
            # build the list of class ids per label via majority vote
            # zoom prob_class to img_shape
            # prob_class_up = zoom(prob_class,
            #                      tuple(s2/s1 for s1, s2 in zip(prob_class.shape[:3], img_shape))+(1,),
            #                      order=0)
            # class_id, label_ids = [], []
            # for reg in regionprops(labels):
            #     m = labels[reg.slice]==reg.label
            #     cls_id = np.argmax(np.mean(prob_class_up[reg.slice][m], axis = 0))
            #     class_id.append(cls_id)
            #     label_ids.append(reg.label)
            # # just a sanity check whether labels where in sorted order
            # assert all(x <= y for x,y in zip(label_ids, label_ids[1:]))
            # res_dict.update(dict(classes = class_id))
            # res_dict.update(dict(labels = label_ids))
            # self.p = prob_class_up

            prob_class = np.asarray(prob_class)
            class_id = np.argmax(prob_class, axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))

        return labels, res_dict