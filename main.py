import asyncio
import traceback
from videosdk.agents import Agent, AgentSession, RealTimePipeline, JobContext, RoomOptions, WorkerJob, Options, function_tool
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.plugins.simli import SimliAvatar, SimliConfig
from videosdk.plugins.turn_detector import NamoTurnDetectorV1

from dotenv import load_dotenv
import os
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

@function_tool
def notify_human(caller_number: str):
    """Notify the business owner that a VIP is on the line."""
    print(f"NOTIFICATION: VIP call detected from {caller_number}")
    return "Notification sent to the owner."

# Define the agent's behavior and personality
class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are an AI receptionist for Akapulu AI. Your goal is to answer questions. If the user mentions they are a VIP or a 'partner,' immediately use the notify_human tool to alert the team. Otherwise, be helpful and concise.",
            tools=[notify_human] # Registers your first 'Action'
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! I'm your real-time assistant. How can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye! It was great talking with you!")

async def start_session(context: JobContext):
    # 1. Initialize Namo for snappy turn-taking
    turn_detector = NamoTurnDetectorV1(threshold=0.7)
    # Configure the Gemini model for real-time voice
    model = GeminiRealtime(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        api_key=os.getenv("GOOGLE_API_KEY"),
        config=GeminiLiveConfig(
            voice="Leda",
            response_modalities=["AUDIO"]
        )
    )

    # 2. Initialize the Simli Avatar
    avatar_config = SimliConfig(
        apiKey=os.getenv("SIMLI_API_KEY"), # Get this from your Simli Dashboard
        faceId="0c2b8b04-5274-41f1-a21c-d5c98322efa9", # This is the default "Tina" face ID
        syncAudio=True
    )
    simli_avatar = SimliAvatar(config=avatar_config)

    pipeline = RealTimePipeline(model=model, avatar=simli_avatar, turn_detector=turn_detector)
    session = AgentSession(agent=MyVoiceAgent(), pipeline=pipeline)

    try:
        await context.connect()
        await session.start()
        await asyncio.Event().wait()
    finally:
        await session.close()
        await context.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions()
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    try:
        # Register the agent with a unique ID
        options = Options(
            agent_id="MyTelephonyAgent",  # CRITICAL: Unique identifier for routing
            register=True,  # REQUIRED: Register with VideoSDK for telephony
            max_processes=10,  # Concurrent calls to handle
            host="localhost",
            port=8081,
        )
        job = WorkerJob(entrypoint=start_session, jobctx=make_context, options=options)
        job.start()
    except Exception as e:
        traceback.print_exc()
