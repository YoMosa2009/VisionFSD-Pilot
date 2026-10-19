// Run with: node pi3b/tests/test_dashboard_imu.js
// Execute the real page function; HTML insertion must never receive error text.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const page = fs.readFileSync(path.join(__dirname, "../web/index.html"), "utf8");
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script.replace("STALE_AFTER_PLACEHOLDER", "2"));
const source = script.slice(script.indexOf("  function updateImuHealth("),
                            script.indexOf("  function updatePanels("));
assert.ok(source.length > 0);
const nodes = {};
for (const id of ["imu-status", "imu-diagnostic"]) {
  assert.ok(page.includes('id="' + id + '"'));
  nodes[id] = {textContent: ""};
  Object.defineProperty(nodes[id], "innerHTML", {
    set() { throw new Error("Diagnostics must not be HTML"); }
  });
}
const context = vm.createContext({$: id => nodes[id]});
vm.runInContext(source, context);
const update = context.updateImuHealth;
const hostile = '<img src=x onerror="throw 1"> & USB failed';
update({imu: "OFF", imu_error: hostile});
assert.equal(nodes["imu-diagnostic"].textContent, hostile);
update({imu: "LIVE", imu_error: "", imu_age_s: .02});
assert.equal(nodes["imu-diagnostic"].textContent, "");
assert.match(nodes["imu-status"].textContent, /LIVE.*0\.02 s ago/);
update({imu: "CAL", imu_calibration: 25, imu_hold: "USB WAIT"});
assert.equal(nodes["imu-status"].textContent, "CAL 25%");
assert.equal(nodes["imu-diagnostic"].textContent, "Calibration: USB WAIT");
update({imu: "STALE", imu_age_s: 1.5});
assert.match(nodes["imu-diagnostic"].textContent, /No fresh IMU samples/);
update({imu: "OFF"});
assert.match(nodes["imu-diagnostic"].textContent, /unavailable or disabled/);
update({});
assert.equal(nodes["imu-status"].textContent, "UNKNOWN");
assert.equal(nodes["imu-diagnostic"].textContent, "");
console.log("IMU dashboard rendering checks passed");
