from twilio.rest import Client
from config import TWILIO_AUTH_TOKEN, TWILIO_ACCOUNT_SID

def call():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    call = client.calls.create(
        twiml='<Response><Say>Hey this is Vedant!</Say></Response>',
        to='+19255776636',
        from_='+18339952474'
        )

    print(call.sid)