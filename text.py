import os
from twilio.rest import Client
from audio import listen

def text(text):
    # Set up the Twilio client
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        to = '+19255776636',
        from_ = '+18339952474', 
        body = text 
    )

    # Print the message SID
    print(message.sid)