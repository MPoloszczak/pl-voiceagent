import logging
from typing import Optional
import asyncio

from agents import Agent, function_tool, ModelSettings, Runner
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from openai.types.responses import ResponseTextDeltaEvent

from utils import logger
from services.mcp_client import tools_for  # NEW

@function_tool
def save_booking_information(date: str, time: str, name: Optional[str] = None, phone: Optional[str] = None, service: Optional[str] = None) -> str:
    """
    Save booking information to the system.
    
    Args:
        date: The date for the booking in YYYY-MM-DD format
        time: The time for the booking in HH:MM format
        name: The customer's name
        phone: The customer's phone number
        service: The service the customer wants to book
    """
    # This is a placeholder. In a real implementation, this would save to a database
    booking_info = {
        "date": date,
        "time": time,
        "name": name,
        "phone": phone,
        "service": service
    }
    logging.info(f"Booking information saved: {booking_info}")
    return f"Successfully saved booking for {date} at {time}"

# Define the booking agent
booking_agent = Agent(
    name="Booking Agent",
    instructions=prompt_with_handoff_instructions(
        """You are a medspa booking assistant. Your goal is to get the necessary information to book a consultation. 
        Get the following information from the user:
        - Preferred date and time for the appointment
        - Name (if possible)
        - Phone number (if possible)
        - Service they're interested in (if possible)
        
        Be friendly, professional, and polite. Stay focused on getting the booking information.
        Once you have the necessary date and time, use the save_booking_information tool to record the booking.
        
        Match the user's tone and communication style to build rapport.
        """
    ),
    model="gpt-4o",
    tools=[save_booking_information]
)

# Define the general conversation agent
general_conversation_agent = Agent(
    name="General Conversation Agent",
    instructions=prompt_with_handoff_instructions(
        """You are a friendly medspa assistant. Your role is to provide information about the medspa, answer questions, 
        and build rapport with potential clients. 
        
        Information about our medspa:
        - We offer a range of services including facials, botox, fillers, laser treatments, and skin consultations
        - We have licensed medical professionals on staff
        - We use only premium products and advanced technology
        - First consultations are complimentary
        
        When responding to the user:
        1. Be warm, friendly, and professional
        2. Match the user's communication style and tone to build rapport
        3. Provide helpful, accurate information about our services
        4. Subtly encourage the user to book a consultation but avoid being pushy
        5. If the user expresses interest in booking, suggest they schedule a consultation
        
        Remember to be conversational, not corporate. Make the user feel valued and understood.
        """
    ),
    model="gpt-4o"
)

# Define the triage agent
triage_agent = Agent(
    name="Triage Agent",
    instructions=prompt_with_handoff_instructions(
        """You are a triage agent for a medspa voice assistant. Your job is to determine the user's intent and route them to the appropriate agent.
        
        If the user's message indicates they want to book an appointment, schedule a consultation, or make a reservation,
        hand off to the Booking Agent who will collect the necessary information.
        
        For all other inquiries, questions, or general conversation, hand off to the General Conversation Agent.
        
        Keep your responses very brief as you are just determining intent, not engaging in the full conversation.
        Only respond if absolutely necessary before making the handoff decision.
        """
    ),
    model="gpt-4o-mini",
    handoffs=[booking_agent, general_conversation_agent]
)

class StreamingHandle:
    """Wrap an async generator to allow cancellation of the LLM stream."""
    def __init__(self, agen):
        self._agen = agen
    def cancel(self):
        """Cancel the async generator to stop the LLM streaming."""
        try:
            aclose = getattr(self._agen, 'aclose', None)
            if aclose:
                # schedule generator close without awaiting
                try:
                    asyncio.create_task(self._agen.aclose())
                except RuntimeError:
                    pass
        except Exception:
            pass

async def get_agent_response(transcript, call_conversation_history, tenant_id: str):
    """
    Process transcripts with the agent and generate a response.
    
    Args:
        transcript: The transcript text from the user
        call_conversation_history: The conversation history for this call
        tenant_id: The ID of the tenant

    Returns:
        tuple: (updated_history, response_text)
    """
    logger.info(f"Running triage agent with input: {transcript}")
    
    # Load dynamic tools per tenant and update booking agent
    try:
        booking_agent.tools = await tools_for(tenant_id)
    except Exception as e:
        logger.error(f"Failed to fetch tools for tenant {tenant_id}: {e}")

    # Create input with conversation history
    agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
    
    # Run the agent
    result = await Runner.run(triage_agent, agent_input)
    
    # Get the agent response
    agent_response = result.final_output
    
    # Return updated history and response
    return result.to_input_list(), agent_response 

async def stream_agent_deltas(transcript: str, call_conversation_history: list, tenant_id: str):
    """
    Stream the agent response and expose a cancellation handle.

    Returns
    -------
    tuple(updated_history, async_generator, handle)
        updated_history : list           Conversation history ready for next turn
        async_generator : AsyncGenerator  Yields raw text deltas for TTS
        handle           : StreamingHandle object with .cancel() used by VAD
    """
    # Load dynamic tools for tenant (for completeness even though general agent may not use them)
    try:
        booking_agent.tools = await tools_for(tenant_id)
    except Exception as e:
        logger.error(f"Failed to fetch tools for tenant {tenant_id}: {e}")

    # Create input with conversation history and user message
    agent_input = call_conversation_history + [{"role": "user", "content": transcript}]
    # Run agent in streaming mode
    result_streaming = Runner.run_streamed(general_conversation_agent, agent_input)
    # Capture updated history for next turn
    updated_history = result_streaming.to_input_list()
    # Async generator for text deltas
    async def delta_generator():
        async for event in result_streaming.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta
    # instantiate generator and wrap it in a handle for cancellation
    agen = delta_generator()
    handle = StreamingHandle(agen)
    return updated_history, agen, handle