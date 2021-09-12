# MidiListener

This is a mini library as a small wrapper around [pyrtmidi](https://github.com/patrickkidd/pyrtmidi) that allows you to instantiate a midi listener and pass it a configuration of callbacks to use in conjunction with other real-time code.

Action config is the critical piece of information that must be passed to MidiListener upon instantiation, or it will be initiated as blank and can be created with createInputHandler functions.

It is a dictionary that is the root for event handling based on controller input. It logs in inputs and the actions they should perform. Here is an example implementation:

```python
# Midi Listener Configuration

# Example functions
def handleKnobInput(velocity):
    print("|" * velocity);

def handleNoteInput(isOn, velocity = None):
    if (velocity):
        return print("noteIsOn:", isOn, velocity);
    else:
        return print("noteIsOn:", isOn);

# Example model
actionConfig = {
    48: {
        'isController': True,
        'callback': handleKnobInput
    },
    52: {
        'isController': True,
        'callback': handleKnobInput
    },
    "D3": {
        'isController': False,
        'callback': handleNoteInput
    }
}
```

# Instantiation

Pass the model file and ensure all functions are in scope to an instantiated midi listener object:

```python
from MidiListener import MidiListener

# actionConfig = { ... }

# Pass the action config and an optional second boolean for verbose logging
listener = MidiListener(actionConfig, True)

listener.start()
```

To add in listeners at runtime, use the `createInputHandler()` helper function. Here's an example of handler creation that creates listeners for the first two knobs on an Akai APC45, and the "D3" keyboard note.

```python
createInputHandler(48, True, handleFirstKnobInput)
createInputHandler(52, True, handleFirstKnobInput)
createInputHandler("D3", False, handleNoteInput)
```
