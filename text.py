from twilio.rest import Client
from audio import listen
from config import TWILIO_AUTH_TOKEN, TWILIO_ACCOUNT_SID, TEST_NUMBER, TWIILIO_NUMBER

def text(text):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    message = client.messages.create(
        to = TEST_NUMBER,
        from_ = TWIILIO_NUMBER, 
        body = text 
    )

    # Print the message SID
    print(message.sid)