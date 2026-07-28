// VisionFSD Pi autonomy firmware for OSOYOO Model 3 V1.1/V1.2 motor shield.
//
// The Arduino is deliberately small and deterministic:
// - motion expires after 350 ms unless the Pi refreshes it;
// - a static, forward ultrasonic sensor blocks forward travel under 18 cm;
// - only the Pi performs LiDAR/camera planning;
// - no Servo scan is attached, reducing continuous battery draw.
//
// Commands at 115200 baud: F, B, L, R, STOP, SPEED 0..105, PING.
// Status: STATUS motion=<F|B|L|R|S> front_cm=<cm|NO_ECHO>

const byte M1_A = 7;       // IN1 (K1/K2)
const byte M1_B = 8;       // IN2
const byte M2_A = 9;       // IN3 (K3/K4)
const byte M2_B = 10;      // IN4
const byte M1_ENABLE = 5;  // ENA
const byte M2_ENABLE = 6;  // ENB
const byte ULTRASONIC_TRIGGER = 3;
const byte ULTRASONIC_ECHO = 2;

const unsigned long COMMAND_TIMEOUT_MS = 350UL;
const unsigned long STATUS_PERIOD_MS = 500UL;
const unsigned long ECHO_TIMEOUT_US = 26000UL;
const int FORWARD_STOP_DISTANCE_CM = 18;
const int DEFAULT_SPEED = 70;
const int MAX_SAFE_SPEED = 105;

char commandBuffer[24];
byte commandLength = 0;
char activeMotion = 'S';
int driveSpeed = DEFAULT_SPEED;
unsigned long lastMotionCommandAt = 0;
unsigned long lastStatusAt = 0;

void setMotor(byte pinA, byte pinB, byte enablePin, int direction, int speed) {
  if (direction == 0 || speed == 0) {
    digitalWrite(pinA, LOW);
    digitalWrite(pinB, LOW);
    analogWrite(enablePin, 0);
    return;
  }
  digitalWrite(pinA, direction > 0 ? HIGH : LOW);
  digitalWrite(pinB, direction > 0 ? LOW : HIGH);
  analogWrite(enablePin, speed);
}

void stopMotors() {
  setMotor(M1_A, M1_B, M1_ENABLE, 0, 0);
  setMotor(M2_A, M2_B, M2_ENABLE, 0, 0);
  activeMotion = 'S';
}

long frontDistanceCentimetres() {
  digitalWrite(ULTRASONIC_TRIGGER, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIGGER, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIGGER, LOW);
  unsigned long pulse = pulseIn(ULTRASONIC_ECHO, HIGH, ECHO_TIMEOUT_US);
  return pulse == 0 ? -1 : static_cast<long>(pulse / 58UL);
}

bool forwardIsBlocked() {
  long distance = frontDistanceCentimetres();
  return distance > 0 && distance < FORWARD_STOP_DISTANCE_CM;
}

void drive(char motion) {
  if (motion == 'F' && forwardIsBlocked()) {
    stopMotors();
    Serial.println(F("BLOCKED:FRONT_ULTRASONIC"));
    return;
  }
  switch (motion) {
    case 'F':
      setMotor(M1_A, M1_B, M1_ENABLE, +1, driveSpeed);
      setMotor(M2_A, M2_B, M2_ENABLE, +1, driveSpeed);
      break;
    case 'B':
      setMotor(M1_A, M1_B, M1_ENABLE, -1, driveSpeed);
      setMotor(M2_A, M2_B, M2_ENABLE, -1, driveSpeed);
      break;
    case 'L':
      setMotor(M1_A, M1_B, M1_ENABLE, -1, driveSpeed);
      setMotor(M2_A, M2_B, M2_ENABLE, +1, driveSpeed);
      break;
    case 'R':
      setMotor(M1_A, M1_B, M1_ENABLE, +1, driveSpeed);
      setMotor(M2_A, M2_B, M2_ENABLE, -1, driveSpeed);
      break;
    default:
      stopMotors();
      return;
  }
  activeMotion = motion;
  lastMotionCommandAt = millis();
}

void handleCommand(const char* command) {
  if (strcmp(command, "STOP") == 0) {
    stopMotors();
  } else if (strcmp(command, "PING") == 0) {
    Serial.println(F("PONG"));
  } else if (strncmp(command, "SPEED ", 6) == 0) {
    driveSpeed = constrain(atoi(command + 6), 0, MAX_SAFE_SPEED);
  } else if (strlen(command) == 1 && strchr("FBLR", command[0]) != NULL) {
    drive(command[0]);
  } else if (command[0] != '\0') {
    Serial.println(F("ERROR:UNKNOWN_COMMAND"));
  }
}

void readSerial() {
  while (Serial.available()) {
    char value = static_cast<char>(Serial.read());
    if (value == '\r') continue;
    if (value == '\n') {
      commandBuffer[commandLength] = '\0';
      handleCommand(commandBuffer);
      commandLength = 0;
    } else if (commandLength < sizeof(commandBuffer) - 1) {
      commandBuffer[commandLength++] = value;
    } else {
      commandLength = 0;  // malformed command: discard safely.
    }
  }
}

void reportStatus() {
  long distance = frontDistanceCentimetres();
  Serial.print(F("STATUS motion="));
  Serial.print(activeMotion);
  Serial.print(F(" front_cm="));
  if (distance < 0) Serial.println(F("NO_ECHO"));
  else Serial.println(distance);
}

void setup() {
  pinMode(M1_A, OUTPUT); pinMode(M1_B, OUTPUT); pinMode(M1_ENABLE, OUTPUT);
  pinMode(M2_A, OUTPUT); pinMode(M2_B, OUTPUT); pinMode(M2_ENABLE, OUTPUT);
  pinMode(ULTRASONIC_TRIGGER, OUTPUT); pinMode(ULTRASONIC_ECHO, INPUT);
  digitalWrite(ULTRASONIC_TRIGGER, LOW);
  stopMotors();
  Serial.begin(115200);
  Serial.println(F("VISIONFSD_PI_AUTONOMY_READY"));
  Serial.println(F("SAFETY:350MS_TIMEOUT,STATIC_FRONT_STOP_18CM,MAX_PWM_105"));
}

void loop() {
  readSerial();
  if (activeMotion != 'S' && millis() - lastMotionCommandAt > COMMAND_TIMEOUT_MS) {
    stopMotors();
    Serial.println(F("STOP:COMMAND_TIMEOUT"));
  }
  if (millis() - lastStatusAt >= STATUS_PERIOD_MS) {
    lastStatusAt = millis();
    reportStatus();
  }
}
