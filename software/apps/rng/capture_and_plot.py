import serial
import matplotlib.pyplot as plt
import sys

PORT = '/dev/tty.usbmodem1102'
BAUD = 38400

print(f"Attempting to connect to {PORT}...")
print("Please ensure you have closed any running miniterm sessions (Ctrl+C or Ctrl+])!")

try:
    ser = serial.Serial(PORT, BAUD, timeout=2)
except Exception as e:
    print(f"Failed to open port: {e}")
    sys.exit(1)

print("\nConnection successful!")
print("Now, please press the Reset button on the back of your Micro:bit to restart the program...")

data = []
started = False

while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if not line:
        continue
        
    if "Board started!" in line:
        print("Detected boot signal! Collecting 10,000 random numbers...")
        started = True
        data = []
        continue
        
    if started:
        if "Finished" in line:
            print("Successfully collected 10,000 numbers!")
            break
            
        try:
            val = int(line)
            data.append(val)
            if len(data) % 1000 == 0:
                print(f"Collected: {len(data)} / 10000...")
        except ValueError:
            pass

ser.close()

# 1. Save to data.txt
with open("data.txt", "w") as f:
    for val in data:
        f.write(f"{val}\n")
print("\nSuccessfully saved all data to data.txt!")

# 2. Automatically plot Histogram
print("Plotting histogram...")
plt.figure(figsize=(10, 6))
plt.hist(data, bins=16, color='skyblue', edgecolor='black')
plt.title("RNG Histogram (10,000 samples, 16 bins)")
plt.xlabel("Random Value")
plt.ylabel("Frequency")
plt.axhline(625, color='red', linestyle='dashed', linewidth=2, label='Expected Frequency (625)')
plt.legend()

# Save as image
plt.savefig("histogram.png")
print("Chart saved as histogram.png!")
plt.show()
