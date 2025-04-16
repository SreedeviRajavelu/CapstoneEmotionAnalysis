#include <Wire.h>
#include <Adafruit_MAX1704X.h>   // Library for MAX17048 fuel gauge
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

#define BUTTON_A 9  // A button is connected to digital pin 9

// Create objects for the MAX17048 fuel gauge and OLED display
Adafruit_MAX17048 maxlipo;
Adafruit_SH1107 display = Adafruit_SH1107(64, 128, &Wire);

unsigned long displayOnTimestamp = 0;
bool displayForcedOn = false;
int lastButtonState = HIGH;  // With INPUT_PULLUP, unpressed reads HIGH

// Battery parameters for estimated runtime calculation (example values)
const float batteryCapacityMah = 10050.0;  // Battery capacity in mAh
const float dischargeCurrentMa = 85.0;     // Average discharge current in mA

// Function to update the OLED display based on the current toggle state
void updateDisplay() {
  if (displayForcedOn) {
    // Retrieve the battery state-of-charge (SOC)
    float soc = maxlipo.cellPercent();

    // Clip soc at 100% if it exceeds 100
    if (soc > 100.0) {
      soc = 100.0;
    }

    float socFraction = soc / 100.0;
    float remainingMah = batteryCapacityMah * socFraction;
    float estimatedRuntime = remainingMah / dischargeCurrentMa;
    
    display.clearDisplay();
    display.setTextColor(SH110X_WHITE);
    
    // Display the battery percentage in text size 2 at the top
    display.setTextSize(2);
    display.setCursor(0, 0);
    display.print(soc, 1);
    display.print("%");  // Printed immediately after to close the gap
    
    // Display the estimated runtime on the next line using text size 1 (starting at y = 30)
    display.setTextSize(1);
    display.setCursor(0, 30);
    display.print("Est Runtime: ");
    display.print(estimatedRuntime, 1);
    display.println("hrs");
    
    // If SOC is below 5%, display "CHARGE!" blinking at y = 40
    if (soc < 5.0) {
      display.setTextSize(2);
      display.setCursor(0, 40);
      bool blink = ((millis() / 500) % 2) == 0;  // Blink every 500 ms
      if (blink) {
        display.println("CHARGE!");
        Serial.println("Battery low: CHARGE! displayed.");
      }
    }
    display.display();
  } else {
    display.clearDisplay();
    display.display();
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  Serial.println("Feather RP2040 Battery Monitor Toggle Demo");
  
  // Initialize I²C
  Wire.begin();
  
  // Initialize the MAX17048 fuel gauge
  if (!maxlipo.begin()) {
    Serial.println("Failed to initialize MAX17048. Check wiring and battery connection!");
    while (1) delay(10);
  }
  Serial.println("MAX17048 initialized successfully.");
  
  // Initialize the OLED display at I²C address 0x3C
  if (!display.begin(0x3C, true)) {
    Serial.println("OLED not found at 0x3C!");
    while (1) delay(10);
  }
  display.setRotation(1);           // Use landscape mode
  display.setContrast(255);           // Maximum contrast
  display.invertDisplay(false);       // Normal (non-inverted) mode
  display.clearDisplay();
  display.display();
  
  // Configure button A as input with the internal pull-up resistor
  pinMode(BUTTON_A, INPUT_PULLUP);
  
  // Initially update the OLED (it starts visible)
  updateDisplay();
  Serial.println("Setup complete. OLED starts visible.");
}

void loop() {
  // Simple button polling: if the button is pressed (reading LOW), toggle the state
  if (digitalRead(BUTTON_A) == LOW) {
    // Brief delay for debounce
    delay(50);
    // Check again to confirm the button is still pressed
    if (digitalRead(BUTTON_A) == LOW) {
      displayForcedOn = !displayForcedOn;  // Toggle the display state
      Serial.print("Button toggled. New OLED state: ");
      Serial.println(displayForcedOn ? "Visible" : "Hidden");
      updateDisplay();
      
      // Wait until the button is released to avoid multiple toggles
      while (digitalRead(BUTTON_A) == LOW) {
        delay(10);
      }
      delay(50); // Additional debounce delay after release
    }
  }
  delay(50);  // Overall loop delay to reduce CPU load
}
