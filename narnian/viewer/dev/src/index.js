import React from 'react';
import ReactDOM from 'react-dom/client';
import Main from "./Main";
import './index.css'  // needed to import Tailwind-related stuff

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <Main />  // "Main" is the base React component associated to the index page (it includes other components)
);
