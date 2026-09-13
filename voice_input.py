import speech_recognition as sr

from processor import process_transaction


def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=1
        )

        audio = recognizer.listen(source)

    print("Processing speech...")

    try:
        text = recognizer.recognize_google(
            audio,
            language="hi-IN"
        )

        print("You said:", text)

        return text

    except sr.UnknownValueError:
        print("Could not understand the speech.")
        return None

    except sr.RequestError as e:
        print("Speech recognition service error:", e)
        return None


def process_voice_transaction():
    text = speech_to_text()

    if text is None:
        return {
            "status": "ERROR",
            "message": "Could not understand speech"
        }

    return process_transaction(text)