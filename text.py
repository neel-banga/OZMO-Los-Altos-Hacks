from twilio.rest import Client
from audio import listen
from config import TWILIO_AUTH_TOKEN, TWILIO_ACCOUNT_SID

def text(text):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    message = client.messages.create(
        to = '+19255776636',
        from_ = '+18339952474', 
        body = text 
    )

    # Print the message SID
    print(message.sid)