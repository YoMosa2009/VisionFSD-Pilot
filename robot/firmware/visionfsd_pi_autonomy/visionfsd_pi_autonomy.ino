// VisionFSD Pi autonomy firmware for OSOYOO Model 3 V1.1/V1.2 motor shield.
//
// The Arduino is deliberately small and deterministic:
// - motion expires after 350 ms unless the Pi refreshes it;
// - a static, forward ultrasonic sensor blocks forward travel under 18 cm;
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
// 12 ms bounds the echo wait to roughly 2 m.  The guard only cares about the
// near field, and a shorter timeout keeps this out of the motor command path.
const unsigned long ECHO_TIMEOUT_US = 12000UL;
// HC-SR04 style sensors need the previous burst to decay before the next one.
// Sampling faster is what produces phantom short readings, and a phantom short
// reading cuts the motors mid-drive.
const unsigned long ULTRASONIC_PERIOD_MS = 60UL;
const int NO_ECHO_CM = 999;
const int FORWARD_STOP_DISTANCE_CM = 18;
const byte BLOCK_CONFIRM_SAMPLES = 2;
// The L298N bridge on this shield drops roughly 2 V, so a 7.9 V pack puts at
// most about 5.6 V across a motor at full duty.  A 105 cap meant 41% of that,
// near 2.3 V, which spins a free wheel in the air but cannot move the robot on
// a floor: the motor just sits energised and buzzing.  The Pi still decides the
// actual speed; this only stops the firmware from being the limit.
const int DEFAULT_SPEED = 118;
const int MAX_SAFE_SPEED = 255;
// Direct PWM under this floor buzzes a loaded wheel instead of moving it.  Do
// not synthesize slow motion by repeatedly pulsing it on/off: that creates the
// exact audible, stop-start behaviour this robot must avoid.
const int MIN_EFFECTIVE_PWM = 105;
const unsigned long OUTPUT_PERIOD_MS = 20UL;
const int OUTPUT_RAMP_STEP = 4;

char commandBuffer[24];
byte commandLength = 0;
char activeMotion = 'S';
int driveSpeed = DEFAULT_SPEED;
int leftOutput = 0;
int rightOutput = 0;
int desiredLeft = 0;
int desiredRight = 0;
unsigned long lastOutputAt = 0;
unsigned long lastMotionCommandAt = 0;
unsigned long lastStatusAt = 0;

int ultrasonicSamples[3] = {NO_ECHO_CM, NO_ECHO_CM, NO_ECHO_CM};
byte ultrasonicIndex = 0;
byte ultrasonicFilled = 0;
int filteredFrontCm = NO_ECHO_CM;
byte blockedStreak = 0;
unsigned long lastUltrasonicAt = 0;

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

int normaliseDrive(int value) {
  if (value == 0) return 0;
  int magnitude = max(MIN_EFFECTIVE_PWM, abs(value));
  return value > 0 ? magnitude : -magnitude;
}

int rampOutput(int current, int target) {
  if (target == 0) return 0;
  target = normaliseDrive(target);
  if (current == 0) return target > 0 ? MIN_EFFECTIVE_PWM : -MIN_EFFECTIVE_PWM;
  if ((current > 0) != (target > 0)) return 0;
  if (abs(target) <= abs(current)) return target;
  int magnitude = min(abs(target), abs(current) + OUTPUT_RAMP_STEP);
  return target > 0 ? magnitude : -magnitude;
}

void writeOutputs() {
  setMotor(M1_A, M1_B, M1_ENABLE, leftOutput > 0 ? +1 : (leftOutput < 0 ? -1 : 0), abs(leftOutput));
  setMotor(M2_A, M2_B, M2_ENABLE, rightOutput > 0 ? +1 : (rightOutput < 0 ? -1 : 0), abs(rightOutput));
}

void updateOutputs() {
  if (millis() - lastOutputAt < OUTPUT_PERIOD_MS) return;
  lastOutputAt = millis();
  leftOutput = rampOutput(leftOutput, desiredLeft);
  rightOutput = rampOutput(rightOutput, desiredRight);
  writeOutputs();
}

void stopMotors() {
  desiredLeft = 0;
  desiredRight = 0;
  leftOutput = 0;
  rightOutput = 0;
  writeOutputs();
  activeMotion = 'S';
}

int measureFrontCentimetres() {
  digitalWrite(ULTRASONIC_TRIGGER, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIGGER, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIGGER, LOW);
  unsigned long pulse = pulseIn(ULTRASONIC_ECHO, HIGH, ECHO_TIMEOUT_US);
  if (pulse == 0) return NO_ECHO_CM;
  int centimetres = static_cast<int>(pulse / 58UL);
  // Under about 2 cm the sensor is inside its own blind zone, so treat that as
  // an invalid sample rather than as an imminent collision.
  return centimetres < 2 ? NO_ECHO_CM : centimetres;
}

int medianOfThree(int a, int b, int c) {
  if (a > b) { int t = a; a = b; b = t; }
  if (b > c) { int t = b; b = c; c = t; }
  if (a > b) { int t = a; a = b; b = t; }
  return b;
}

// Sampled on a fixed cadence from loop(), never from inside a DRIVE command.
// Previously the ping happened both per command and per status report, so two
// could land milliseconds apart; the second then read the first burst's echo
// and reported a phantom short distance, which cut the motors mid-drive.
void updateUltrasonic() {
  ultrasonicSamples[ultrasonicIndex] = measureFrontCentimetres();
  ultrasonicIndex = (ultrasonicIndex + 1) % 3;
  if (ultrasonicFilled < 3) ultrasonicFilled++;
  filteredFrontCm = ultrasonicFilled < 3
      ? NO_ECHO_CM
      : medianOfThree(ultrasonicSamples[0], ultrasonicSamples[1], ultrasonicSamples[2]);
  if (filteredFrontCm < FORWARD_STOP_DISTANCE_CM) {
    if (blockedStreak < BLOCK_CONFIRM_SAMPLES) blockedStreak++;
  } else {
    blockedStreak = 0;
  }
}

bool forwardIsBlocked() {
  return blockedStreak >= BLOCK_CONFIRM_SAMPLES;
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
  // Any command with a forward component is gated.  A tight arc idles its
  // inner wheel, giving values like (200, 0); testing for "both wheels
  // positive" would let exactly those close-quarters turns past the guard.
  // Reverse and pivots have a negative wheel and stay ungated, so the Pi can
  // always drive itself out of a tight spot.
  bool forwardComponent = (left >= 0 && right >= 0) && (left > 0 || right > 0);
  if (forwardComponent && forwardIsBlocked()) {
    stopMotors();
    Serial.println(F("BLOCKED:FRONT_ULTRASONIC"));
    return;
  }
  desiredLeft = normaliseDrive(left);
  desiredRight = normaliseDrive(right);
  activeMotion = describeMotion(left, right);
  lastMotionCommandAt = millis();
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
  // Reports the filtered value and never pings here: the sampler owns the
  // cadence.  The PWM fields are printed unconditionally, so a NO_ECHO status
  // no longer silently drops them.
  if (filteredFrontCm >= NO_ECHO_CM) Serial.print(F("NO_ECHO"));
  else Serial.print(filteredFrontCm);
  Serial.print(F(" left_pwm="));
  Serial.print(leftOutput);
  Serial.print(F(" right_pwm="));
  Serial.print(rightOutput);
  Serial.print(F(" blocked="));
  Serial.println(forwardIsBlocked() ? 1 : 0);
}

void setup() {
  pinMode(M1_A, OUTPUT); pinMode(M1_B, OUTPUT); pinMode(M1_ENABLE, OUTPUT);
  pinMode(M2_A, OUTPUT); pinMode(M2_B, OUTPUT); pinMode(M2_ENABLE, OUTPUT);
  pinMode(ULTRASONIC_TRIGGER, OUTPUT); pinMode(ULTRASONIC_ECHO, INPUT);
  digitalWrite(ULTRASONIC_TRIGGER, LOW);
  stopMotors();
  Serial.begin(115200);
  Serial.println(F("VISIONFSD_PI_AUTONOMY_READY"));
  Serial.println(F("SAFETY:350MS_TIMEOUT,MEDIAN_FRONT_STOP_18CM,DIRECT_PWM_RAMP,MIN_PWM_105,MAX_PWM_255"));
}

void loop() {
  readSerial();
  if (millis() - lastUltrasonicAt >= ULTRASONIC_PERIOD_MS) {
    lastUltrasonicAt = millis();
    updateUltrasonic();
    // A confirmed close obstacle must also interrupt travel already under way,
    // not only a newly arriving forward command.
    if (activeMotion == 'F' && forwardIsBlocked()) {
      stopMotors();
      Serial.println(F("BLOCKED:FRONT_ULTRASONIC"));
    }
  }
  updateOutputs();
  if (activeMotion != 'S' && millis() - lastMotionCommandAt > COMMAND_TIMEOUT_MS) {
    stopMotors();
    Serial.println(F("STOP:COMMAND_TIMEOUT"));
  }
  if (millis() - lastStatusAt >= STATUS_PERIOD_MS) {
    lastStatusAt = millis();
    reportStatus();
  }
}
