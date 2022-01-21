from pyrt_wrapper.listen import startListening
import threading

# Primary export
class MidiListener:
    def __init__(self, actionConfig={}, verbose=False):
        self.actionConfig = actionConfig
        self.verbose = verbose
        self.running_thread = None
        self.stop_threads = False

    def start(self, options={}):
        self.running_thread = threading.Thread(
            target=startListening,
            args=(options, self.actionConfig, lambda: self.stop_threads),
        )
        # startListening(options, self.actionConfig)
        self.running_thread.start()

    def stop(self, options={}):
        self.stop_threads = True
        self.running_thread.join()

    # createInputHandler is an unnecessary helper function to create config options on the go, but could be useful to surface in the API
    # if this is ever a package and is imported elsewhere
    def createInputHandler(self, noteName, isController, handlerCallback):
        self.actionConfig[noteName] = {
            "isController": isController,
            "callback": handlerCallback,
        }

    def deleteInputHandler(self, noteName):
        del self.actionConfig[noteName]
