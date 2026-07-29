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
const unsigned long ECHO_TIMEOUT_US = 26000UL;
const int FORWARD_STOP_DISTANCE_CM = 18;
// The L298N bridge on this shield drops roughly 2 V, so a 7.9 V pack puts at
// most about 5.6 V across a motor at full duty.  A 105 cap meant 41% of that,
// near 2.3 V, which spins a free wheel in the air but cannot move the robot on
// a floor: the motor just sits energised and buzzing.  The Pi still decides the
// actual speed; this only stops the firmware from being the limit.
const int DEFAULT_SPEED = 150;
const int MAX_SAFE_SPEED = 255;

// Static friction needs more torque to break than motion needs to sustain, so
// a start from rest gets a brief full-power pulse before settling to the
// commanded value.  Without it a low cruise command can never get going.
const unsigned long KICKSTART_MS = 90UL;
const int KICKSTART_PWM = 220;

char commandBuffer[24];
byte commandLength = 0;
char activeMotion = 'S';
int driveSpeed = DEFAULT_SPEED;
int leftOutput = 0;
int rightOutput = 0;
int desiredLeft = 0;
int desiredRight = 0;
unsigned long kickstartUntil = 0;
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

// Applies the commanded outputs, boosted while a kickstart pulse is active.
// Called both when a command arrives and from loop(), so the boost ends on
// time rather than lasting until the next command happens to turn up.
void applyOutputs() {
  int left = desiredLeft;
  int right = desiredRight;
  if (millis() < kickstartUntil) {
    if (left != 0 && abs(left) < KICKSTART_PWM) left = left > 0 ? KICKSTART_PWM : -KICKSTART_PWM;
    if (right != 0 && abs(right) < KICKSTART_PWM) right = right > 0 ? KICKSTART_PWM : -KICKSTART_PWM;
  }
  setMotor(M1_A, M1_B, M1_ENABLE, left > 0 ? +1 : (left < 0 ? -1 : 0), abs(left));
  setMotor(M2_A, M2_B, M2_ENABLE, right > 0 ? +1 : (right < 0 ? -1 : 0), abs(right));
  leftOutput = left;
  rightOutput = right;
}

void stopMotors() {
  desiredLeft = 0;
  desiredRight = 0;
  kickstartUntil = 0;
  applyOutputs();
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

char describeMotion(int left, int right) {
  if (left == 0 && right == 0) return 'S';
  if (left >= 0 && right >= 0) return 'F';
  if (left <= 0 && right <= 0) return 'B';
  return left < right ? 'L' : 'R';
}

void driveDifferential(int left, int right) {
  left = constrain(left, -MAX_SAFE_SPEED, MAX_SAFE_SPEED);
  right = constrain(right, -MAX_SAFE_SPEED, MAX_SAFE_SPEED);
  if (left > 0 && right > 0 && forwardIsBlocked()) {
    stopMotors();
    Serial.println(F("BLOCKED:FRONT_ULTRASONIC"));
    return;
  }
  // Breaking away from rest, or reversing a wheel, is where torque is scarce.
  bool wasStopped = (desiredLeft == 0 && desiredRight == 0);
  bool reversed = (desiredLeft > 0) != (left > 0) || (desiredRight > 0) != (right > 0);
  if ((left != 0 || right != 0) && (wasStopped || reversed)) {
    kickstartUntil = millis() + KICKSTART_MS;
  }
  desiredLeft = left;
  desiredRight = right;
  applyOutputs();
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
  long distance = frontDistanceCentimetres();
  if (distance < 0) Serial.println(F("NO_ECHO"));
  else {
    Serial.print(distance);
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
  Serial.println(F("VISIONFSD_PI_AUTONOMY_READY"));
  Serial.println(F("SAFETY:350MS_TIMEOUT,STATIC_FRONT_STOP_18CM,DIFFERENTIAL_DRIVE,MAX_PWM_105"));
}

void loop() {
  readSerial();
  // Ends the kickstart pulse on schedule instead of at the next command.
  if (activeMotion != 'S') {
    applyOutputs();
  }
  if (activeMotion != 'S' && millis() - lastMotionCommandAt > COMMAND_TIMEOUT_MS) {
    stopMotors();
    Serial.println(F("STOP:COMMAND_TIMEOUT"));
  }
  if (millis() - lastStatusAt >= STATUS_PERIOD_MS) {
    lastStatusAt = millis();
    reportStatus();
  }
}
