from twilio.rest import Client
from config import TWILIO_AUTH_TOKEN, TWILIO_ACCOUNT_SID, TEST_NUMBER, TWIILIO_NUMBER

def call():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    call = client.calls.create(
        twiml='<Response><Say>Hey this is Vedant!</Say></Response>',
        to=TEST_NUMBER,
        from_=TWIILIO_NUMBER
        )

    print(call.sid)