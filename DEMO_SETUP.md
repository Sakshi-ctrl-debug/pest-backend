# 🚀 PEST DETECTION BACKEND - DEMO SETUP GUIDE

## ⚡ Quick Start (For Tomorrow's Demo)

### **Option 1: Automatic Start (EASIEST - Use This!)**
1. Open Windows Explorer
2. Navigate to: `C:\pest app\pests_detection\project\pest_detection_backend\`
3. **Double-click** `START_BACKEND.bat`
4. Wait until you see:
   ```
   🌐 Starting FastAPI server...
   📍 Backend will be available at: http://192.168.1.105:8000
   ```
5. Leave this window open during your demo
6. Your app will work immediately!

---

### **Option 2: Manual Start (If batch file doesn't work)**
1. Open PowerShell
2. Run these commands:
   ```powershell
   cd "C:\pest app\pests_detection\project\pest_detection_backend"
   .\venv\Scripts\Activate.ps1
   uvicorn main:app --host 192.168.1.105 --port 8000
   ```
3. You should see similar output as Option 1

---

## ✅ Verify Backend is Running

### **Test 1: Browser Test (Easiest)**
1. Open your phone or laptop browser
2. Go to: `http://192.168.1.105:8000/health`
3. You should see:
   ```json
   {
     "status": "healthy",
     "message": "Backend is running and ready for pest detection",
     "model": "pest_model.h5",
     "classes": 5
   }
   ```

### **Test 2: API Docs (Advanced)**
1. Go to: `http://192.168.1.105:8000/docs`
2. You should see an interactive API documentation page

### **Test 3: In the Flutter App**
1. Open the pest detection app
2. Toggle to **Network Mode** (switch OFF)
3. Pick an image
4. Click "Detect Pest"
5. You should see the result in 2-4 seconds

---

## 🔧 Troubleshooting

### **Problem: "Virtual environment not found"**
**Solution:**
```powershell
cd "C:\pest app\pests_detection\project\pest_detection_backend"
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```
Then try again.

### **Problem: "Port 8000 already in use"**
**Solution:** Kill the existing process:
```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```
Then restart the backend.

### **Problem: "Module not found" error**
**Solution:** Reinstall dependencies:
```powershell
cd "C:\pest app\pests_detection\project\pest_detection_backend"
.\venv\Scripts\pip install --upgrade -r requirements.txt
```

### **Problem: "pest_model.h5 not found"**
**Solution:** Make sure these files are in the backend folder:
- `pest_model.h5` (the trained model)
- `class_names.json` (the class labels)
- `pest_info.py` (pest information)

### **Problem: Model loads but crashes on detection**
**Solution:** Check that your phone is on the same Wi-Fi as your PC
- Phone: Settings → Wi-Fi → Connect to same network as PC
- PC: Should have IP `192.168.1.105`

---

## 📋 Pre-Demo Checklist

- [ ] Backend folder exists: `C:\pest app\pests_detection\project\pest_detection_backend\`
- [ ] All 3 files present: `pest_model.h5`, `class_names.json`, `pest_info.py`
- [ ] Virtual environment exists: `venv\` folder
- [ ] START_BACKEND.bat exists
- [ ] Test health check: `http://192.168.1.105:8000/health`
- [ ] Phone is on same Wi-Fi network as PC
- [ ] Flutter app is installed on phone with **Network Mode** working
- [ ] Backend starts without errors

---

## 🎯 Demo Day Setup (Step-by-Step)

### **Morning of Demo:**
1. **Start Backend** (5 minutes)
   - Double-click `START_BACKEND.bat`
   - Wait for "Backend is ready" message
   - Keep window open

2. **Start Flutter App** (2 minutes)
   - Open app on phone
   - Check it's on same Wi-Fi

3. **Quick Test** (2 minutes)
   - Pick an image
   - Test Network Mode detection
   - Verify results appear

4. **Start Demo**
   - Show the app to your college
   - Pick sample images
   - Click detect
   - Show results!

---

## 🚨 Emergency Fixes

### **If backend crashes during demo:**
```powershell
# Kill the process
taskkill /F /IM python.exe

# Wait 5 seconds
Start-Sleep -Seconds 5

# Restart
cd "C:\pest app\pests_detection\project\pest_detection_backend"
.\venv\Scripts\Activate.ps1
uvicorn main:app --host 192.168.1.105 --port 8000
```

### **If app won't connect:**
1. Check phone is on same Wi-Fi
2. Check PC IP is still 192.168.1.105:
   ```powershell
   ipconfig /all | findstr "192.168"
   ```
3. Test backend: `http://192.168.1.105:8000/health`
4. Restart backend if needed

---

## 📞 Quick Reference

| Need | Command |
|------|---------|
| Start Backend | Double-click `START_BACKEND.bat` |
| Check IP | `ipconfig /all` |
| Check Port | `netstat -ano | findstr :8000` |
| Kill Process | `taskkill /F /IM python.exe` |
| Install Deps | `pip install -r requirements.txt` |
| Test Health | Visit `http://192.168.1.105:8000/health` in browser |

---

## 💡 Pro Tips

1. **Don't close the backend window** during your demo
2. **Test beforehand** - run through the demo once the night before
3. **Have phone on silent** to avoid distractions
4. **Use Network Mode** for demo (toggles off) - it's more impressive and accurate
5. **Keep venv activated** if manually starting
6. **Keep internet active** - Wi-Fi network needs to stay running

---

## ✨ You're All Set!

Your backend is production-ready. Just:
1. Double-click `START_BACKEND.bat` tomorrow
2. Keep it running during your demo
3. Enjoy showing off your Pest Detection App! 🎉

Good luck with your college presentation!
