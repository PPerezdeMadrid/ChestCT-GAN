var fs = require('fs');
var https = require('https');
var express = require('express');
var session = require('express-session');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var createError = require('http-errors');

var indexRouter = require('./routes/index');
var loginRouter = require('./routes/login');
var contactRouter = require('./routes/contact');
var proyectoRouter = require('./routes/proyecto');
var profileRouter = require('./routes/profile');
var lungCTRouter = require('./routes/lungCT');
var adminContactRouter = require('./routes/adminContact');
var adminBlogRouter = require('./routes/adminBlog');
var adminUsersRouter = require('./routes/adminUsers');
var blogRouter = require('./routes/blog');
var entradaRouter = require('./routes/entrada');

var app = express();

// Cargar certificados SSL (autofirmados o de una CA)
const options = {
    key: fs.readFileSync('/etc/letsencrypt/live/chestgan.goduck.org/privkey.pem'),
    cert: fs.readFileSync('/etc/letsencrypt/live/chestgan.goduck.org/fullchain.pem')
  };

// Configurar las sesiones
app.use(session({
  secret: 'CLAVE', 
  resave: false,
  saveUninitialized: true
}));

app.use('/.well-known', express.static('/var/www/html/.well-known'));

const createDatabase = require('./modules/database');
createDatabase();

// View engine setup
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/login', loginRouter);
app.use('/contact', contactRouter);
app.use('/proyecto', proyectoRouter);
app.use('/lungCT', checkAuthenticated, lungCTRouter);
app.use('/profile', checkAuthenticated, profileRouter);
app.use('/adminContact', checkAdmin, adminContactRouter);
app.use('/adminBlog', checkAdmin, adminBlogRouter);
app.use('/adminUsers', checkAdmin, adminUsersRouter);
app.use('/blog', blogRouter);
app.use('/entrada', entradaRouter);

function checkAuthenticated(req, res, next) {
  if (!req.session.user) {  
    return res.redirect('/login'); 
  }
  next();  
}

function checkAdmin(req, res, next) {
  if (!req.session.user || !req.session.user.isAdmin) {  
    return res.redirect('/login'); 
  }
  next();
}

app.get('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) console.log("Ha habido un error: ", err);
    console.log("El usuario ha cerrado la sesión");
    res.redirect('/'); 
  });
});

app.use((req, res, next) => {
  next(createError(404));
});

app.use((err, req, res, next) => {
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};
  res.status(err.status || 500);
  res.render('error');
});

// Crear servidor HTTPS
https.createServer(options, app).listen(443, () => {
  console.log('Servidor HTTPS corriendo en el puerto 443');
});

module.exports = app;

/*
 sudo certbot certonly --webroot -w /var/www/html -d chestgan.duckdns.org
*/
