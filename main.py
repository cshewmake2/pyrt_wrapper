from listen import startListening

# Primary export
class MidiListener():
    actionConfig = {}

    def __init__(self, actionConfig = {}, verbose = False):
        self.actionConfig = actionConfig;
        self.verbose = verbose

    def start(self, options = {}):
        startListening(options, self.actionConfig)

    # createInputHandler is an unnecessary helper function to create config options on the go, but could be useful to surface in the API 
    # if this is ever a package and is imported elsewhere
    def createInputHandler(self, noteName, isController, handlerCallback):
        self.actionConfig[noteName] = {
            'isController': isController,
            'callback': handlerCallback
        }
    def deleteInputHandler(self, noteName):
        del self.actionConfig[noteName]