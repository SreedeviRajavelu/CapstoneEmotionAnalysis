# ignore this file, use the other try_server2 .py instead 

import socketio
import asyncio
from aiohttp import web
import time
import threading
import queue

# Assuming the previous script is imported as data_processor
from mahata_integration import DataProcessor, get_latest_prediction

# Create a Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

class StreamServer:
    def __init__(self, mac_address):
        # Create data processor
        self.processor = DataProcessor(mac_address)
        
        # Start data processing thread
        self.data_thread = threading.Thread(
            target=self.processor.main_loop, 
            daemon=True
        )
        self.data_thread.start()

    async def send_updates(self):
        while True:
            try:
                # Get latest prediction
                prediction = get_latest_prediction(self.processor)
                
                if prediction:
                    # Emit prediction to all connected clients
                    print(f"Prediction: {prediction}")
                    prediction_update = f"{prediction['stress']},{prediction['valence']}"
                    await sio.emit('StressValenceUpdate', prediction_update)
                    print(f"Sent update: {prediction}")
                
                # Wait for a bit before next update
                await asyncio.sleep(2)
            
            except Exception as e:
                print(f"Error in update loop: {e}")
                await asyncio.sleep(2)

async def main():
    # MAC address of your device
    mac_address = "98:D3:91:FD:40:9B"
    
    # Create stream server
    stream_server = StreamServer(mac_address)
    
    # Create task for periodic updates
    asyncio.create_task(stream_server.send_updates())

    # Setup web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 5000)
    await site.start()

    print("Server is running on http://0.0.0.0:5000")
    
    # Keep the server running
    while True:
        await asyncio.sleep(3600)

# Run the server
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.")
