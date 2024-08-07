#include <Brain.h>


Brain brain(Serial);

int pushButton =2;

void setup() {
  Serial.begin(9600);
  pinMode(pushButton, INPUT);
}

void loop() { 
  int buttonState = digitalRead(pushButton);
  
  if (brain.update()) {
    String brainData = brain.readCSV();
    String output = brainData + "," + buttonState;
    Serial.println(output);

  }
}
