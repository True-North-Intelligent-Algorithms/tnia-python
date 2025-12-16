---
marp: true
paginate: true
backgroundImage: url('logo.jpeg')
backgroundSize: 25%
backgroundPosition: 5% 95%
---

# Pixi and Napari-Easy-Augment-Batch-DL  

_An agile approach for reproducible and repeatable deep learning_

---

# Why Pixi?  Why environments ? 

Deep learning is 90% Python environment management

* Open Source (Cellpose GUI/Napari Plugins/Fiji Plugins) are written in Python or Python Wrappers
* Fiji can now call into Python environments 
* Many (most?) Commercial products are Cellpose and Stardist Python wrappers
* Call into Python 'Environment' 
* For repeatable workflows need to know what environment and version you are using
---

# Why Pixi?  Why environments ? 

* Pixi makes environments easier
* expert can set up Pixi configuration files
* beginner can run one command to setup environment and run an application

---

# Repeatable vs Reproducible  software 

1.  Repeatable - Same software same data same results
2.  Reproducible level 1 - different software, same data, same conclusions
3.  Reproducible level 2 - different software, different data, same conclusions

---

# Pixi and Napari for repeatable and reproducible deep learning

* Pixi helps us get the same versions of dependencies all the time!  
* Napari (with plugins like Napari-Easy-Augment-Batch-DL) helps us run deep learning frameworks and test if they are reproducible between each other

----

# Goals

1.  Show how to use Pixi to start a Napari plugin with different deep learning frameworks.
2.  Use Napari Layers to compare results from the different frameworks.  Are they reproducible?

----

# Disclaimer
- I am not an expert in Pixi
- Pixi has power-law productivity
- Early knowledge yields large productivity benefit

---

# Disclaimer Napari-Easy-Augment-Batch-DL
- Under Construction
<img src="under_construction.png" alt="Under Construction" width="300"/>

---
## The Project: napari‑easy‑augment‑batch‑dl
- A plugin for **napari** to augment data and apply deep learning segmentation
- Wraps multiple deep learning frameworks including **Cellpose** and **MicroSAM**
- **Combine multiple deep‑learning toolkits in the same workflow**
- Advanced options are **very configurable** for special cases
- Designed for rapid experimentation:
  - Add new augmentations quickly
  - Test less commonly used options 

---
```
## Agile by Design
- We make changes **very fast**
- Small user base = **less risk of breaking** existing workflows
- Contrast:
  - **Cellpose GUI** & **QuPath** → large user bases (thousands), stable APIs
  - Move slower, break less
- napari‑easy‑augment‑batch‑dl → perfect for trying **special options fast**

---

## Why Agile Matters Here
✅ Quick iterations on new augmentation ideas  
✅ Feedback loops directly with early users  
✅ Can integrate new segmentation toolkits or new versions of existing toolkits ASAR (As soon as Released)  
✅ Can respond to specific research needs without delay  

---

## Big Tools vs. Niche Tools
**Cellpose GUI / QuPath:**
- Work great ~80% of the time
- Large communities, stability first

**napari‑easy‑augment‑batch‑dl:**
- Niche, but powerful for the last 20%
- When you need an obtuse setting or option it can be ready in hours. 
- Experimental features can be 'released' immediately

---

## Pixi as the Perfect Partner
- **Pixi** sets up consistent environments quickly
    - fine grained control over exact version of dependency toolkits
- You can even pin to a **specific commit** of napari‑easy‑augment‑batch‑dl
- Ideal for agile workflows:
  - Test a commit
  - Roll back
```
---

## Example Workflow
1. Navigate to folder with 'pixi.toml'
2. Navigate to prefonfigured pixi project
3. `pixi run startup`

## Preconfigured Pixi projects
1. Napari-easy-augment-batch + Stardist
2. Napari-easy-augment-batch + Cellpose/Microsam

---

## When to Choose napari-easy-augment-batch-dl
✅ Need special augmentation options  
✅ Willing to try cutting‑edge features  
✅ Don’t mind quick changes and updates  
✅ Already using **Cellpose** or **MicroSAM** and want to push them further

---

## Questions?
💬 Let’s talk about Pixi, agile workflows, and how to get the most from **napari‑easy‑augment‑batch‑dl**!

