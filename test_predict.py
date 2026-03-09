"""Quick prediction test script to verify the /predict endpoint works correctly."""
import glob, json, sys, urllib.request

# Find a test image
imgs = glob.glob("dataset/**/*.jpg", recursive=True) or \
       glob.glob("dataset/**/*.jpeg", recursive=True) or \
       glob.glob("dataset/**/*.png", recursive=True)

if not imgs:
    print("ERROR: No images found in dataset/")
    sys.exit(1)

img_path = imgs[0]
print(f"Testing with: {img_path}")

# POST to /predict
boundary = "testboundary123"
with open(img_path, "rb") as f:
    img_data = f.read()

body = (
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
    f"filename=\"test.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode()
    + img_data
    + f"\r\n--{boundary}--\r\n".encode()
)

req = urllib.request.Request(
    "http://127.0.0.1:5000/predict",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

try:
    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)
    print("\n=== PREDICTION RESULT ===")
    print(f"Disease raw    : {result.get('disease')}")
    print(f"Display name   : {result.get('display_name')}")
    print(f"Confidence     : {result.get('confidence')}%")
    print(f"Severity       : {result.get('severity')}")
    print(f"Description    : {result.get('tips', {}).get('description', '')[:80]}...")
    print(f"\nSeverity is _default? : {result.get('severity') == 'unknown'}")
except Exception as e:
    print(f"ERROR: {e}")
