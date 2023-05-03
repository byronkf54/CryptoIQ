const express = require('express');
const path = require('path');
const React = require('react');
const ReactDOMServer = require('react-dom/server');
const HomePage = require('./Client/home.js');

const app = express();
const port = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
  const app = ReactDOMServer.renderToString(React.createElement(HomePage));
  res.send(`
    <html>
      <head>
        <title>Crypto Rating</title>
      </head>
      <body>
        <div id="app">${app}</div>
        <script src="/bundle.js"></script>
      </body>
    </html>
  `);
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
