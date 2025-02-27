**How to setup the DEV environment to work on the viewer**

- Install Node.js from https://nodejs.org/en/download.

- Goto your home folder (or wherever you like). Assuming you are in your home folder ~, then:


```
mkdir .npm-global                    
npm config set prefix '~/.npm-global'
export PATH=~/.npm-global/bin:$PATH
```

- Install some dependecies:

```
npm install -g create-react-app
npm install -D tailwindcss@3 postcss autoprefixer
npm install d3
npm install graphlib-dot
npm install react-plotly.js plotly.js
npm install framer-motion 
npm install lucide-react
npm install react-dnd
npm install react-dnd-html5-backend
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

- Goto the project folder and then the viewer subfolder

```
cd <narnian project folder>
cd narnian
cd viewer
```

*[PERSONAL NOTE - DON'T DO THIS] If you are creating a new app from scratch, bootstrap it with (otherwise dont' do it!!!!):*
*npx create-react-app viewer;*
*npx tailwindcss init*

- Run the development server:

```
npm start
```

- You are now able to access the API calls to http://localhost:3000/

- You can convert your website to a static site offered with the usual Web Servers (such as Apache) by running (it will be moved to the "www" folder that is in "viewer")

```
npm run build
```

- Now you can close the **npm start** thing, since the whole website and rest APIs will be offered by the internal Python server at http://localhost:5000/

