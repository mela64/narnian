**What is this?**

This is the folder which contains the web interface to visualize a running environment.

It consists of a React app, and, as usual, it allows to work in **DEV** mode and **PRODUCTION** mode.

If you need to make changes to the interface, be sure to be in **DEV** mode. 
This can be achieved by following the instruction below.

Otherwise, no need to do anything, the **PRODUCTION**-mode interface is in the **www** folder and it will
be automatically offered by our Python server at http://localhost:5001/.


**How to setup the DEV mode to work on the interface (to edit/change it)**

- Install Node.js from https://nodejs.org/en/download.

- Goto your home folder (or wherever you like). 

- Assuming you are in your home folder ~, then:

```
mkdir .npm-global                    
npm config set prefix '~/.npm-global'
export PATH=~/.npm-global/bin:$PATH
```

- Install some dependencies:

```
npm install -D tailwindcss@3 postcss autoprefixer
npm install d3
npm install graphlib-dot
npm install react-plotly.js plotly.js
npm install framer-motion 
npm install lucide-react
npm install react-dnd
npm install react-dnd-html5-backend
npm install tailwind-scrollbar-hide
```

- Fix/update packages:

```
npm audit
npm audit fix --force
```

- Do it again (repeat until you see the smallest number of critical issues, warning: it has a loopy behaviour):

```
npm audit fix --force
```

- *Ok, in the future you can skip all the steps above, and start from here!*
- Goto the project folder and then the viewer subfolder

```
cd <narnian project folder>
cd narnian
cd viewer
```

- Run the development server:

```
npm start
```

- You are now able to access the *DEV* interface at http://localhost:3000/

**Are you done with your changes? Do you want to switch to PRODUCTION MODE?**

- Close the **npm start** thing, since, when the **PRODUCTION** mode will be active, 
the whole website and rest APIs will be 
offered by the internal Python server at http://localhost:5001/

- You can now convert your website to, say, a classic site by running 
(it will be moved to the **www** folder that is in "viewer")

```
npm run build
```
- This will delete and re-populate the **www** folder with the update interface, 
and you are now in **PRODUCTION** mode.

