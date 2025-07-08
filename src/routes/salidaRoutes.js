const express = require('express');
const router = express.Router();
const salidaController = require('../controllers/salidaController');

// Ruta POST para crear salida
router.post('/ingreso', salidaController.crearSalida);

// Ruta GET para obtener lista de salidas
router.get('/lista', salidaController.obtenerSalidas);

// Ruta GET para obtener productos
router.get('/productos', salidaController.obtenerProductos);

// Ruta GET para buscar cliente por DNI
router.get('/cliente/:dni', salidaController.buscarClientePorDNI);

// Ruta POST para generar minería de datos
router.post('/mineria', salidaController.generarMineria);

module.exports = router;