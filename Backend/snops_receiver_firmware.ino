#include <SPI.h>
#include <LoRa.h>

// LoRa pins
#define SS 15
#define RST 16
#define DI0 5

// Sensor data structure to match Python expectations
struct SensorData {
  float temp;
  float hum;
  int gas;
  int vib;
  int dist;
};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("SNOPS LoRa Receiver Starting...");

  // Initialize LoRa
  LoRa.setPins(SS, RST, DI0);
  
  if (!LoRa.begin(915E6)) {
    Serial.println("LoRa init failed. Check your connections!");
    while (1);
  }
  
  LoRa.setSyncWord(0xF1);
  Serial.println("LoRa init succeeded.");
  Serial.println("Waiting for data...");
}

void loop() {
  // Try to parse packet
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    Serial.print("Received packet: ");
    
    // Read packet
    String received = "";
    while (LoRa.available()) {
      received += (char)LoRa.read();
    }
    
    Serial.println(received);
    
    // Parse the received data (assuming it's in JSON format from transmitter)
    // For now, just forward it to serial
    Serial.println(received);
  }
  
  delay(100);
}
