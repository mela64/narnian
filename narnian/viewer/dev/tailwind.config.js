/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,js,jsx,ts,tsx}",  // Adjust this to point to your source files
  ],
  theme: {
    extend: {},
    screens: {
      sm: '640px', // Small screen (phones)
      md: '768px', // Medium screen (tablets)
      lg: '1024px', // Large screen (desktops)
      xl: '1280px', // Extra large screen
    },
  },
  plugins: [],
};
