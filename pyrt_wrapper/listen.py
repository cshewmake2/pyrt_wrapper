import rtmidi

midiin = rtmidi.RtMidiIn()


def returnHandlerIfExists(midi, actionConfig):
    if midi.isController():
        controllerNumber = midi.getControllerNumber()
        return (
            actionConfig[controllerNumber]
            if (controllerNumber in actionConfig)
            else False
        )
    else:
        note = midi.getMidiNoteName(midi.getNoteNumber())
        return actionConfig[note] if (note in actionConfig) else False


# Global controller to handle midi actions coming in if they are present
def handleMidiAction(midi, actionConfig, verbose):
    eventHandler = returnHandlerIfExists(midi, actionConfig)

    if verbose:
        if midi.isController():
            print("Received input from Controller: ", midi.getControllerNumber())
        else:
            print("Received input from:", midi.getMidiNoteName(midi.getNoteNumber()))
        print("Has event handler? ", bool(eventHandler))

    # If there is no handler for this note, do nothing.
    if bool(eventHandler) == False:
        return

    if eventHandler["isController"]:
        # print("CONTROLLER", midi.getControllerNumber(), midi.getControllerValue())
        eventHandler["callback"](midi.getControllerValue())

    else:
        if midi.isNoteOn():
            velocity = midi.getVelocity()
            print(
                "ON: ", midi.getMidiNoteName(midi.getNoteNumber()), midi.getVelocity()
            )
            eventHandler["callback"](True, velocity)
        else:
            print("OFF:", midi.getMidiNoteName(midi.getNoteNumber()))
            eventHandler["callback"](False)


def startListening(options, actionConfig, stop):
    # Port listening setup
    ports = range(midiin.getPortCount())
    isVerbose = "verbose" in options

    if ports:
        for i in ports:
            print(midiin.getPortName(i))
        print("Opening port 0!")
        midiin.openPort(0)
        while not stop():
            m = midiin.getMessage(250)  # some timeout in ms
            if m:
                handleMidiAction(m, actionConfig, isVerbose)
    else:
        print("NO MIDI INPUT PORTS!")
