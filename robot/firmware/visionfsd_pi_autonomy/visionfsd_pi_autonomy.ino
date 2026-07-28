// VisionFSD Pi autonomy firmware for OSOYOO Model 3 V1.1/V1.2 motor shield.
//
// The Arduino is deliberately small and deterministic:
// - motion expires after 350 ms unless the Pi refreshes it;
// - a static, forward ultrasonic sensor blocks forward travel under 18 cm;
// - target PWM is ramped in small steps, avoiding abrupt current spikes and
//   the stop-start feel caused by immediate motor-output changes;
// - only the Pi performs LiDAR/camera planning;
// - no Servo scan is attached, reducing continuous battery draw.
//
// Commands at 115200 baud: F, B, L, R, STOP, SPEED 0..105,
// DRIVE <left PWM> <right PWM>, PING.  DRIVE values are -105..105.
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
const unsigned long STATUS_PERIOD_MS = 250UL;
const unsigned long ULTRASONIC_PERIOD_MS = 60UL;
const unsigned long RAMP_PERIOD_MS = 20UL;
const unsigned long ECHO_TIMEOUT_US = 26000UL;
const int FORWARD_STOP_DISTANCE_CM = 18;
const int DEFAULT_SPEED = 70;
const int MAX_SAFE_SPEED = 105;
const int PWM_RAMP_STEP = 7;

char commandBuffer[24];
byte commandLength = 0;
char activeMotion = 'S';
int driveSpeed = DEFAULT_SPEED;
int leftOutput = 0;       // actual PWM currently applied to the motors
int rightOutput = 0;
int targetLeftOutput = 0;
int targetRightOutput = 0;
long latestFrontCm = -1;
bool frontBlocked = false;
unsigned long lastMotionCommandAt = 0;
unsigned long lastStatusAt = 0;
unsigned long lastUltrasonicAt = 0;
unsigned long lastRampAt = 0;

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
  leftOutput = 0;
  rightOutput = 0;
  targetLeftOutput = 0;
  targetRightOutput = 0;
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

void sampleUltrasonic() {
  latestFrontCm = frontDistanceCentimetres();
  // A close valid echo stops immediately.  No echo is deliberately not
  // treated as a false obstacle; the Pi's LD19 remains the wider safety view.
  frontBlocked = latestFrontCm > 0 && latestFrontCm < FORWARD_STOP_DISTANCE_CM;
}

char describeMotion(int left, int right) {
  if (left == 0 && right == 0) return 'S';
  if (left >= 0 && right >= 0) return 'F';
  if (left <= 0 && right <= 0) return 'B';
  return left < right ? 'L' : 'R';
}

void driveDifferential(int left, int right) {
  left = constrain(left, -MAX_SAFE_SPEED, MAX_SAFE_SPEED);
  right = constrain(right, -MAX_SAFE_SPEED, MAX_SAFE_SPEED);
  if (left > 0 && right > 0 && frontBlocked) {
    stopMotors();
    Serial.println(F("BLOCKED:FRONT_ULTRASONIC"));
    return;
  }
  targetLeftOutput = left;
  targetRightOutput = right;
  activeMotion = describeMotion(left, right);
  lastMotionCommandAt = millis();
}

int rampTowards(int current, int target) {
  if (current == target) return current;
  // Finish braking before changing direction, protecting the gearbox and
  // avoiding the sharp current reversal that made this chassis feel jerky.
  if ((current > 0 && target < 0) || (current < 0 && target > 0)) {
    target = 0;
  }
  if (current < target) return min(current + PWM_RAMP_STEP, target);
  return max(current - PWM_RAMP_STEP, target);
}

void applyMotorOutputs() {
  setMotor(M1_A, M1_B, M1_ENABLE, leftOutput > 0 ? +1 : (leftOutput < 0 ? -1 : 0), abs(leftOutput));
  setMotor(M2_A, M2_B, M2_ENABLE, rightOutput > 0 ? +1 : (rightOutput < 0 ? -1 : 0), abs(rightOutput));
}

void updateMotorRamp() {
  if (millis() - lastRampAt < RAMP_PERIOD_MS) return;
  lastRampAt = millis();
  leftOutput = rampTowards(leftOutput, targetLeftOutput);
  rightOutput = rampTowards(rightOutput, targetRightOutput);
  applyMotorOutputs();
}

void drive(char motion) {
  switch (motion) {
    case 'F':
      driveDifferential(driveSpeed, driveSpeed);
      break;
    case 'B':
      driveDifferential(-driveSpeed, -driveSpeed);
      break;
    case 'L':
      driveDifferential(-driveSpeed, driveSpeed);
      break;
    case 'R':
      driveDifferential(driveSpeed, -driveSpeed);
      break;
    default:
      stopMotors();
      return;
  }
}

void handleCommand(const char* command) {
  if (strcmp(command, "STOP") == 0) {
    stopMotors();
  } else if (strcmp(command, "PING") == 0) {
    Serial.println(F("PONG"));
  } else if (strcmp(command, "CAPS") == 0) {
    Serial.println(F("CAPS DRIVE"));
  } else if (strncmp(command, "SPEED ", 6) == 0) {
    driveSpeed = constrain(atoi(command + 6), 0, MAX_SAFE_SPEED);
  } else if (strncmp(command, "DRIVE ", 6) == 0) {
    char* end = NULL;
    long left = strtol(command + 6, &end, 10);
    if (end == command + 6) {
      Serial.println(F("ERROR:DRIVE_LEFT"));
      return;
    }
    while (*end == ' ') ++end;
    char* rightEnd = NULL;
    long right = strtol(end, &rightEnd, 10);
    if (rightEnd == end || *rightEnd != '\0') {
      Serial.println(F("ERROR:DRIVE_RIGHT"));
      return;
    }
    driveDifferential(static_cast<int>(left), static_cast<int>(right));
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
  Serial.print(F("STATUS motion="));
  Serial.print(activeMotion);
  Serial.print(F(" front_cm="));
  if (latestFrontCm < 0) Serial.println(F("NO_ECHO"));
  else {
    Serial.print(latestFrontCm);
    Serial.print(F(" left_pwm="));
    Serial.print(leftOutput);
    Serial.print(F(" right_pwm="));
    Serial.println(rightOutput);
  }
}

void setup() {
  pinMode(M1_A, OUTPUT); pinMode(M1_B, OUTPUT); pinMode(M1_ENABLE, OUTPUT);
  pinMode(M2_A, OUTPUT); pinMode(M2_B, OUTPUT); pinMode(M2_ENABLE, OUTPUT);
  pinMode(ULTRASONIC_TRIGGER, OUTPUT); pinMode(ULTRASONIC_ECHO, INPUT);
  digitalWrite(ULTRASONIC_TRIGGER, LOW);
  stopMotors();
  Serial.begin(115200);
  sampleUltrasonic();
  Serial.println(F("VISIONFSD_PI_AUTONOMY_READY"));
  Serial.println(F("SAFETY:350MS_TIMEOUT,STATIC_FRONT_STOP_18CM,DIFFERENTIAL_DRIVE,MAX_PWM_105"));
}

void loop() {
  readSerial();
  if (millis() - lastUltrasonicAt >= ULTRASONIC_PERIOD_MS) {
    lastUltrasonicAt = millis();
    sampleUltrasonic();
  }
  if (frontBlocked && targetLeftOutput > 0 && targetRightOutput > 0) {
    stopMotors();
    Serial.println(F("BLOCKED:FRONT_ULTRASONIC"));
  }
  if (activeMotion != 'S' && millis() - lastMotionCommandAt > COMMAND_TIMEOUT_MS) {
    stopMotors();
    Serial.println(F("STOP:COMMAND_TIMEOUT"));
  }
  updateMotorRamp();
  if (millis() - lastStatusAt >= STATUS_PERIOD_MS) {
    lastStatusAt = millis();
    reportStatus();
  }
}
