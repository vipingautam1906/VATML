# VATML: Ventricular Arrhythmia Detection using TinyML #
Cardiac arrhythmias pose a significant health risk, underscoring the critical need for precise detection. Introducing "VATML: Ventricular Arrhythmia Detection using TinyML" demonstrating TinyML's potential to enable detection within implantable cardioverter defibrillators (ICDs). VATML features a lightweight 1D Convolutional Neural Network (CNN), tailored for life-threatening Ventricular Arrhythmias (VAs), with a minimal 15.86 KB footprint and low Multiply-Accumulate (MAC) complexity of 12.08K, meeting ICD operational demands.  

## Features
- Lightweight 1D CNN for Ventricular Arrhythmia Detection
- Minimal footprint and low MAC complexity
- Compatible with implantable cardioverter defibrillators (ICDs)
![plot](/images/flow.png)

## Have a try ##

### Set up environment
- cd ./Python_source
- pip install -r requirements.txt

 
### Training and Testing
- python train.py 
- python test.py 

### Deployment on Nucleo STM32f303k8 board

Export the model from .pkl to onnx 
```
    python pkl2onnx.py
```
For CubeMX  

-  We have shared the pretrained weights (./Weights/model_best.onnx), import them from ./Weights/model_best.onnx  in network option, select Compression: None (default)  
-  In advanced settings, click on ONLY "Use activation buffer for input buffer" & "Use activation buffer for the output buffer" 

For Keil-MDK5

- Option for Target->Target->Code generation：Use default compiler version 6 & Use Micro LIB;
- Option for Target->C/C++(AC6)->Optimization: -Oz image size
- Option for Target->C/C++(AC6)->Click on "One ELF Section per Function" & "Link Time Optimization" & "Execute-only Code" & "Short enums/wchar"
- Load the model on STM32F303K8 

## Results on Nucleo STM32f303k8 board
<img src="/images/Table1.png" alt="Setup Image" width="500"/> <img src="/images/lineplot.png" alt="Flow Image" width="600"/>
<img src="/images/Table2.png" alt="Setup Image" width="1200"/>



  


