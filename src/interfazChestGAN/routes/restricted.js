var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('restricted', { username: req.session.username });
});

module.exports = router;
