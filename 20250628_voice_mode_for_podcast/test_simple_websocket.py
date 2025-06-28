#!/usr/bin/env python3
"""
Simple WebSocket test with hardcoded token
"""

import asyncio
import websockets
import json

# Use the token we got from curl
TOKEN = "AQICAHg2rXGpktPz9xtZE1fXlazdxmaeKoy12AkCobXSonBt6AFMdjaik0awsIGE9AmlwWDvAAADkjCCA44GCSqGSIb3DQEHBqCCA38wggN7AgEAMIIDdAYJKoZIhvcNAQcBMB4GCWCGSAFlAwQBLjARBAyHYle_edoGItO-AdwCARCAggNFdauvHN4rPByFhkqjCizAR-hJZGdfGQPCyVkY3XB5QyINeGx75i1ZNJYKoYIYh6qVhjs2EhLXclILZ4CA4NuoBTuSOevdRJ4GG8AJJdhHr-UP7R3LX7XShk6BFbSr-yMRciYh5ed8eZtj2FpveulPcTvuUpVYvyaBVZt07M_-7X5N8bQwlHwSrtpPbrVHqB3Z5Uz_a3HzQLPGKNl-r8kwGgQ0vxBntGqOY01s70K_WQGju3vYF7uOc_WKEuaBndVARkEL_Xtgh3AjSo15xNmgDgFYwn1TyYpp3qHVM5G8tQfaKBlFPYf7fVBzcZuJmIxGkfxkUW4ejOxKD8-46U1pIIOTkLIO9XwJtPzhtV3X8ieuIBixK8_hdsVjU2hghOzhWOGhi3KRY1Vffm58zzUdOxecn1bmbpQ8JTNn9uIxYR19qZxuqAGhraqDNuV0dIHuKjKG-KUZAjhH4lCkxTWLcIiJvPaNvnOkiWfi7Kpv6fmyMBGEgGpGubtFHd9S5TAjZAXEW22zdCRjbCaKBu2to9QpKi1UhquB898uXDx8l6QAKLP5Bmig81eLVoKtIEo2VzTrYH2gWVe1bxxdJyN5vfjJPua2Kgcm-lI2s-JvJ1Ob_zy34-ZhgMATgrBgqqpMfTDnB6vhd1wyYxQT3wPqLBfL0SANHamRaVE7VbUrUpkBtjr36cK3NIFOnpC6bx26RORU6afzJ57ywveCkSPfmj04irwGG5H2iv75022N0wUms5gGeumlnGnW8cwfsTSpvUM67rRVaYg9UX8jbA1tmxkc4UQ3jN3y_fYvAj5GAW-CesUqC01osTXkw7s51m5rB7pUS6X9duhNfvgqox3hjbJmUwkX5Z_-yyA-bSafjyBQcJB3C3oi8vTo_z-Qf4TqhHf4Mi8qZcGfVzHbjTPhj-bxyWdvcYrvQa87jZl5N32XCPAgPSSY5OqVwJfgnBCOqS0fZSy7Szg6f1Qe3xA6bg6hMz07yaSt0PgdN_c6dQFEO3ig-M3u0CAST5iOXpB9qKZJqf0f0nTmWDNbl4amMw3p4mtZxk5gZJp8iSvxi1yMdqhWTFQu3F5xMvAF7E3CnWQhH9c31J7JI4i5TA6WjikX2wDn"

async def test_websocket():
    uri = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&encoding=pcm_s16le&format_turns=true"
    
    try:
        print(f"🔗 Connecting to {uri}")
        print(f"🔑 Using token: {TOKEN[:50]}...")
        
        async with websockets.connect(
            uri, 
            additional_headers={"Authorization": f"Bearer {TOKEN}"}
        ) as websocket:
            print("✅ Connected!")
            
            # Listen for messages
            async for message in websocket:
                data = json.loads(message)
                print(f"📨 Received: {data}")
                
                if data.get('type') == 'Begin':
                    print(f"🌊 Session started: {data.get('id')}")
                    break
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
