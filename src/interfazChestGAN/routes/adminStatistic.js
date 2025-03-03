const express = require('express');
const router = express.Router();
const sqlite3 = require('sqlite3').verbose();

const db = new sqlite3.Database('./database.db');

// Ruta para obtener estadísticas
router.get('/', (req, res) => {
    // Obtener las fechas de inicio y fin desde los parámetros de consulta (si existen)
    let startDate = req.query.startDate || '1970-01-01';
    let endDate = req.query.endDate || '9999-12-31'; 

    // Si no se incluyen horas en las fechas, añadirlas al final
    if (startDate.length === 10) {
        startDate += ' 00:00:00';  
    }
    if (endDate.length === 10) {
        endDate += ' 23:59:59';  
    }

    // Consultar número de mensajes de contacto
    db.get("SELECT COUNT(*) AS contact_messages FROM mensajes_contacto", (err, contactMessages) => {
        if (err) {
            console.error(err);
            return res.status(500).send("Error al obtener datos de mensajes de contacto");
        }

        // Consultar número de publicaciones en el blog
        db.get("SELECT COUNT(*) AS publications FROM entradas_blog", (err, publications) => {
            if (err) {
                console.error(err);
                return res.status(500).send("Error al obtener datos de publicaciones");
            }

            // Consultar número de usuarios
            db.get("SELECT COUNT(*) AS users FROM usuarios", (err, users) => {
                if (err) {
                    console.error(err);
                    return res.status(500).send("Error al obtener datos de usuarios");
                }

                // Consultar estadísticas de eval_tomografias con filtro de fechas
                const query = `SELECT * FROM eval_tomografias WHERE created_at BETWEEN ? AND ?`;
                db.all(query, [startDate, endDate], (err, evalData) => {
                    if (err) {
                        console.error(err);
                        return res.status(500).send("Error al obtener datos de eval_tomografias");
                    }

                    // Cálculos adicionales para eval_tomografias
                    const totalResponses = evalData.length;
                    const avgResponse = evalData.reduce((acc, row) => acc + row.response, 0) / totalResponses;

                    // Respuestas reales
                    const realResponses = evalData.filter(row => row.is_real === 1).length;
                    const nonRealResponses = totalResponses - realResponses;

                    // Calcular la puntuación media de las respuestas no reales
                    const nonRealData = evalData.filter(row => row.is_real === 0);
                    const avgNonRealResponse = nonRealData.length > 0 ? nonRealData.reduce((acc, row) => acc + row.response, 0) / nonRealData.length : 0;

                    // Para la distribución de respuestas (gráfico de barras)
                    const responseCounts = Array(10).fill(0); 

                    evalData.forEach(row => {
                        if (row.response >= 1 && row.response <= 10) {
                            responseCounts[row.response - 1]++;
                        }
                    });

                    // Pasar los datos a la vista
                    res.render('adminViews/adminStatistics', {
                        user: req.session.user,
                        contactMessages: contactMessages.contact_messages,
                        publications: publications.publications,
                        users: users.users,
                        avgResponse: avgResponse,
                        realResponses: realResponses,
                        nonRealResponses: nonRealResponses,
                        totalResponses: totalResponses,
                        avgNonRealResponse: avgNonRealResponse,
                        startDate: startDate,
                        endDate: endDate,
                        responseCounts: responseCounts 
                    });
                });
            });
        });
    });
});

module.exports = router;
