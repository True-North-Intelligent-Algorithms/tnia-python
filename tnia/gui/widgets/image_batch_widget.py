# import
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTreeWidget, QTreeWidgetItem, QAbstractItemView
from tnia.gui.widgets.tree_image_item import TreeImageItem
from qtpy.QtCore import QThread

class ImageBatch(QWidget):
    def __init__(self, parent, op, viewer):
        super().__init__(parent)
        self.viewer = viewer

        self.op = op
        
        self.v_layout = QVBoxLayout()
        
        # add a horizontal layout for the two buttons
        # button one adds the active image
        # button two adds all images
        self.h_layout = QHBoxLayout()
        self.add_active_image_button = QPushButton('Add Active Image', self)
        self.add_all_images_button = QPushButton('Add All Images', self)
        self.h_layout.addWidget(self.add_active_image_button)
        self.h_layout.addWidget(self.add_all_images_button)
        
        self.v_layout.addLayout(self.h_layout)
        
        # create the tree widget
        self.treeWidget = QTreeWidget(self)
        self.treeWidget.setColumnCount(3)
        self.treeWidget.setHeaderLabels(["Checked", "Data Name", "Progress"])
        self.v_layout.addWidget(self.treeWidget)

        # add the run buttons (consider putting this in a horizontal layout)
        run_active_button = QPushButton('Run Active', self)
        self.v_layout.addWidget(run_active_button)
        run_batch_button = QPushButton('Run Batch', self)
        self.v_layout.addWidget(run_batch_button)
        
        self.setLayout(self.v_layout)
        
        # accept drops
        self.setAcceptDrops(True)
        self.treeWidget.setDragDropMode(QAbstractItemView.DragDrop)
        self.treeWidget.setAcceptDrops(True)
        self.treeWidget.setDragEnabled(True)

        # connect the tree widget to handlers
        self.treeWidget.dragEnterEvent = self.drag_enter_event
        self.treeWidget.dragMoveEvent = self.dragMoveEvent
        self.treeWidget.dropEvent = self.dropEvent
        self.treeWidget.itemDoubleClicked.connect(self.on_item_double_clicked)

        # connect the buttons to handlers
        self.add_active_image_button.clicked.connect(self.add_active_image_clicked)
        run_active_button.clicked.connect(self.run_active)
        run_batch_button.clicked.connect(self.run_batch)

        self.images = []

    def add_active_image_clicked(self):
        # check if there is an active image
        if self.viewer.layers.selection.active is not None:
            self.images.append(self.viewer.layers.selection.active)
            self.treeWidget.addTopLevelItem(TreeImageItem(self.treeWidget, self.viewer.layers.selection.active.name, self.viewer.layers.selection.active))
    
    def drag_enter_event(self, event):
        event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        print('event.mimeData().urls()', event.mimeData().urls())
        print()
        print('event', event)   

        files = [url.toLocalFile() for url in event.mimeData().urls()]

        for file in files:
            #layers = self.viewer.open(file)
            self.images.append(file)
            # get name of file
            file_name = file.split("/")[-1]

            self.treeWidget.addTopLevelItem(TreeImageItem(self.treeWidget, file_name, file))
            # remove these layers
            #for layer in layers:
            #    self.viewer.layers.remove(layer)


        for item in event.mimeData().items:
            self.treeWidget.addTopLevelItem(TreeImageItem(self.treeWidget, item.name, item))

    def run_active(self):
        print("run active",self.viewer.layers.selection.active.name)

    def run_batch(self):
        print("run batch")
        
        for i in range(self.treeWidget.topLevelItemCount()):
            print('input is ',self.treeWidget.topLevelItem(i).image, 'type is ', type(self.treeWidget.topLevelItem(i).image))
            self.i = i
            self.treeWidget.topLevelItem(i).output = self.op.run(self.treeWidget.topLevelItem(i).image, self.update_progress)
            #for j in range(100):
            #    self.treeWidget.topLevelItem(i).update_progress(j)
            #    # pause 10 milliseconds
            #    QThread.msleep(10)
            #print(self.treeWidget.topLevelItem(i).text(1))

    def update_progress(self, progress):
        self.treeWidget.topLevelItem(self.i).update_progress(progress)

    def on_item_double_clicked(self, item, column):
        # Ensure the item is of type TreeImageItem
        if isinstance(item, TreeImageItem):
            print(f"Item '{item.text(1)}' was double-clicked!")
            # Add any additional response logic here
            print(f"full path: {item.full_path}")
            from skimage.io import imread
            image = imread(item.full_path)
            self.viewer.add_image(image, name=item.text(1))

            if item.output is not None:
                if isinstance(item.output, str):
                    image = imread(item.output)
                    self.viewer.add_image(image, name=item.text(1) + "_out")
