### v.0.1.0

[Fixed bug]
1. Issue on vertical 'U' component waveform (Hi-net dataset)
- predict.py > def wf2matrix()
- [updated]
st=st0.select(channel='*Z')
if len(st)==0:
    st=st0.select(channel='*U')

[Updates]
- New features utilizing the S/P amplitude ratio are updated (Example2).

---

### v.0.0.2

[Fixed bug]
1. Issue on vertical component waveform
- predict.py > def wf2matrix()
- [previous] st.select(channel='*Z')
- [updated] st=st.select(channel='*Z')

[Updates]
- Uploaded pre-trained model files to github repository.

---

### v.0.0.1
- First version of RPNet

---