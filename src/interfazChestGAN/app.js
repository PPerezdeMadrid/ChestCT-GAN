var createError = require('http-errors');
var express = require('express');
var session = require('express-session');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var indexRouter = require('./routes/index');
var loginRouter = require('./routes/login');
var contactRouter = require('./routes/contact');
var proyectoRouter = require('./routes/proyecto');
var profileRouter = require('./routes/profile');


var app = express();

// Configurar las sesiones
app.use(session({
  secret: 'CLAVE',  // Usa una clave secreta para cifrar la sesión
  resave: false,
  saveUninitialized: true
}));

const createDatabase = require('./modules/database');
createDatabase();

// view engine setup
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
app.use('/profile', checkAuthenticated, profileRouter);

// Middleware para verificar si el usuario está logueado
function checkAuthenticated(req, res, next) {
  console.log(req.session); 
  if (!req.session.username) {  
    return res.redirect('/login'); 
  }
  next();  
}

app.get('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      console.log("Ha habido un error: ", err) 
    }
    console.log("El usuario ha cerrado la sesión")
    res.redirect('/'); 
  });
});



module.exports = checkAuthenticated;



module.exports = app;
