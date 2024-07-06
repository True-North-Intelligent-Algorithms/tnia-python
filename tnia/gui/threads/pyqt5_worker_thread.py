
from PyQt5.QtCore import QObject, pyqtSignal

class PyQt5WorkerThread(QObject):
    """
    A worker class that runs a function in a separate thread.

    Attributes:
        finished (pyqtSignal): Signal that is emitted when the function has finished running.
        progress (pyqtSignal): Signal that is emitted to report progress. It emits an integer and a string.
        func (callable): The function to run in the worker thread.
        args (tuple): Positional arguments to pass to `func`.
        kwargs (dict): Keyword arguments to pass to `func`.
        is_cancelled (bool): Flag that indicates whether the worker has been cancelled.
    """

    finished = pyqtSignal()
    progress = pyqtSignal(str, int)
    update_images = pyqtSignal()

    def __init__(self, func, *args, **kwargs):
        """
        Initialize the Worker with the function to run and its arguments.

        Args:
            func (callable): The function to run in the worker thread.
            *args: Positional arguments to pass to `func`.
            **kwargs: Keyword arguments to pass to `func`.
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.is_cancelled = False

    def run(self):
        """
        Run the function with the given arguments. This method is intended to be called in a separate thread.
        """
        self.func(*self.args, **self.kwargs)
        self.finished.emit()
