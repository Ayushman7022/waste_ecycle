Waste Battery vs Wire Classifier
Setup guide for Windows

This program uses your webcam to tell a battery from a wire.
It does not need a dataset and it does not train a model.
The first run downloads the model (about 350 MB). Later runs work offline.


WHAT YOU NEED
-------------
- Windows 10 or Windows 11
- Python 3.10 or newer
  Download from https://www.python.org/downloads/
  During install, tick "Add python.exe to PATH"
- A webcam
- An internet connection for the first run only


1. OPEN THE PROJECT FOLDER
--------------------------
Copy this whole folder to the other computer, then open PowerShell
in that folder. Example:

    cd D:\waste_disposable

If the folder is somewhere else, use that path instead.


2. CREATE A VIRTUAL ENVIRONMENT
-------------------------------
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

If PowerShell says the script is blocked, run this once, then activate again:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned


3. INSTALL THE LIBRARIES
------------------------
    pip install -r requirements.txt

This can take several minutes. Torch and the vision libraries are large.


4. START THE WEB APP
--------------------
    .\.venv\Scripts\python.exe -m uvicorn webapp:app --host 127.0.0.1 --port 8000

Wait until the terminal says the model is ready.
Then open a browser on the SAME computer and go to:

    http://127.0.0.1:8000

You should see the live camera, the current label, and two lists:
Battery and Wire.

Leave the PowerShell window open while you use the page.
Closing the browser tab does not stop the camera.
To stop the program, click the PowerShell window and press Ctrl+C.


5. HOW PHOTOS ARE SAVED
-----------------------
A photo is saved only when a new battery or a new wire appears
and stays in view for about half a second.
Holding the same object does not save another photo.
A cup, phone, or other object lights "invalid" and is not saved.
Take the object away until the page says "none" or "too far",
then show the next object.

Saved photos go into:

    captures\battery\
    captures\wire\

Each file looks like:

    20261005_121430_battery_0.84.jpg

The number at the end is the confidence.
"none", "too far", and "invalid" are not saved.


6. HOW TO USE THE CAMERA
------------------------
- Put the battery or wire in the center of the picture.
- Hold it close, about within one metre.
- Use a plain desk or wall behind it.
- Good light helps.

Labels you may see:

    battery    a battery was found
    wire       a wire or cable was found
    invalid    something else was found; it is not saved
    none       nothing confident was found
    too far    the object looks too small, so it was skipped


7. IF THE WRONG CAMERA OPENS
----------------------------
Stop the server with Ctrl+C, then try camera 1:

    $env:CAMERA = "1"
    .\.venv\Scripts\python.exe -m uvicorn webapp:app --host 127.0.0.1 --port 8000

If that is still wrong, try "2" instead of "1".
Also close Zoom, Teams, or any other app that is using the webcam.
In Windows Settings, allow desktop apps to use the camera.


8. OPTIONAL: OPENCV WINDOW (NO SAVING)
--------------------------------------
This opens a camera window on the desktop. It does not save photos.

    .\.venv\Scripts\python.exe classify.py

Press q or Esc in that window to quit.

Useful options:

    --camera 1         use another webcam
    --threshold 0.70   require higher confidence (default is 0.60)
    --threshold 0.50   accept weaker matches
    --min-size 0.15    require the object to look closer
    --min-size 0.06    allow objects that look a bit farther
    --min-size 0       turn the distance check off

Example:

    .\.venv\Scripts\python.exe classify.py --camera 0 --threshold 0.60 --min-size 0.10


9. NEXT TIME YOU USE IT
-----------------------
You do not need to install again. From the project folder:

    .\.venv\Scripts\Activate.ps1
    .\.venv\Scripts\python.exe -m uvicorn webapp:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000


TROUBLESHOOTING
---------------
"python is not recognized"
    Python is not on PATH. Reinstall Python and tick "Add python.exe to PATH",
    or open a new PowerShell window after installing.

pip fails or times out
    Check the internet connection and run the pip command again.

The page does not open
    The server must still be running in PowerShell.
    Use http://127.0.0.1:8000 on the same PC. This address is not for other computers.

The preview is gray or shows the wrong picture
    Set CAMERA to 1 or 2, as in section 7.

Everything says "none"
    Move the object closer and into the center. Lower the threshold
    only in the OpenCV window, with --threshold 0.50.

Everything says "too far"
    Hold the object closer, or run the OpenCV window with --min-size 0.06.

The first start is slow
    The model is downloading or loading. Wait until the terminal says it is ready.
