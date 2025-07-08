const { PrismaClient } = require('@prisma/client');
const { v4: uuidv4 } = require('uuid');

const prisma = new PrismaClient();

// Constantes
const USUARIO_FIJO = "4fc37618-84be-454b-a4b4-d1231f961737";

// POST - Crear nueva salida
const crearSalida = async (req, res) => {
  try {
    const {
      montoTotal,
      tipoOperacion = "VENTA_NACIONAL_01",
      tipoDocumento = "BOLETA_VENTA_03",
      fechaEntrega,
      estadoEntrega = "PENDIENTE",
      estadoPago = "EN_VERIFICACION", 
      serie = "V001",
      idCliente,
      idProveedor,
      productos // Array de productos: [{ idProducto, cantidad, precioUnitario, descripcion }]
    } = req.body;

    // Validaciones básicas
    if (!montoTotal || !fechaEntrega || !productos || productos.length === 0) {
      return res.status(400).json({
        error: 'Faltan campos obligatorios',
        required: ['montoTotal', 'fechaEntrega', 'productos']
      });
    }

    // Validar que no se envíen cliente y proveedor a la vez
    if (idCliente && idProveedor) {
      return res.status(400).json({
        error: 'No se puede especificar cliente y proveedor al mismo tiempo'
      });
    }

    // Obtener el último número de la serie para incrementar
    const ultimaSalida = await prisma.salida.findFirst({
      where: { serie },
      orderBy: { numero: 'desc' }
    });

    const nuevoNumero = ultimaSalida ? ultimaSalida.numero + 1 : 1;

    // Obtener ubicación del cliente si se especifica
    let idUbicacionEntrega = null;
    if (idCliente) {
      const cliente = await prisma.cliente.findUnique({
        where: { idCliente },
        include: { ubicacion: true }
      });
      
      if (!cliente) {
        return res.status(404).json({ error: 'Cliente no encontrado' });
      }
      
      idUbicacionEntrega = cliente.idUbicacion;
    }

    // Crear la salida con transacción
    const resultado = await prisma.$transaction(async (prisma) => {
      // Crear la salida
      const nuevaSalida = await prisma.salida.create({
        data: {
          idSalida: uuidv4(),
          montoTotal: parseFloat(montoTotal),
          tipoOperacion,
          tipoDocumento,
          fecha: new Date(),
          fechaEntrega: new Date(fechaEntrega),
          estadoEntrega,
          estadoPago,
          numero: nuevoNumero,
          serie,
          idCliente: idCliente || null,
          idProveedor: idProveedor || null,
          idUsuario: USUARIO_FIJO,
          idUbicacionEntrega
        }
      });

      // Crear los productos de la salida
      const productosCreados = await Promise.all(
        productos.map(async (producto) => {
          return await prisma.productoSalida.create({
            data: {
              idProductoSalida: uuidv4(),
              idSalida: nuevaSalida.idSalida,
              idProducto: producto.idProducto,
              descripcion: producto.descripcion || '',
              cantidad: parseInt(producto.cantidad),
              precioUnitario: parseFloat(producto.precioUnitario)
            }
          });
        })
      );

      return { salida: nuevaSalida, productos: productosCreados };
    });

    res.status(201).json({
      success: true,
      message: 'Salida creada exitosamente',
      data: resultado
    });

  } catch (error) {
    console.error('Error al crear salida:', error);
    res.status(500).json({
      error: 'Error interno del servidor',
      message: error.message
    });
  }
};

// GET - Obtener lista de salidas
const obtenerSalidas = async (req, res) => {
  try {
    const salidas = await prisma.salida.findMany({
      include: {
        cliente: {
          select: {
            documento: true,
            nombre: true,
            apellidoPaterno: true,
            apellidoMaterno: true
          }
        },
        proveedor: {
          select: {
            ruc: true,
            razonSocial: true
          }
        },
        usuario: {
          select: {
            nombre: true,
            apellidoPaterno: true,
            apellidoMaterno: true
          }
        },
        productosSalida: {
          include: {
            producto: {
              select: {
                codigo: true,
                nombre: true
              }
            }
          }
        }
      },
      orderBy: {
        fecha: 'desc'
      }
    });

    res.status(200).json({
      success: true,
      count: salidas.length,
      data: salidas
    });

  } catch (error) {
    console.error('Error al obtener salidas:', error);
    res.status(500).json({
      error: 'Error interno del servidor',
      message: error.message
    });
  }
};


// GET - Obtener lista de productos
const obtenerProductos = async (req, res) => {
    try {
      const productos = await prisma.producto.findMany({
        where: {
          estado: true // Solo productos activos
        },
        select: {
          idProducto: true,
          codigo: true,
          nombre: true,
          precioUnitario: true,
          stock: true,
          categoria: {
            select: {
              nombre: true
            }
          }
        },
        orderBy: {
          nombre: 'asc'
        }
      });
  
      res.status(200).json({
        success: true,
        count: productos.length,
        data: productos
      });
  
    } catch (error) {
      console.error('Error al obtener productos:', error);
      res.status(500).json({
        error: 'Error interno del servidor',
        message: error.message
      });
    }
  };
  
  // GET - Buscar cliente por DNI
  const buscarClientePorDNI = async (req, res) => {
    try {
      const { dni } = req.params;
  
      // Validar que el DNI tenga 8 dígitos
      if (!/^\d{8}$/.test(dni)) {
        return res.status(400).json({
          success: false,
          error: 'DNI debe tener exactamente 8 dígitos'
        });
      }
  
      const cliente = await prisma.cliente.findUnique({
        where: {
          documento: dni
        },
        include: {
          ubicacion: {
            include: {
              distrito: {
                include: {
                  provincia: {
                    include: {
                      region: true
                    }
                  }
                }
              }
            }
          }
        }
      });
  
      if (!cliente) {
        return res.status(404).json({
          success: false,
          error: 'Cliente no encontrado',
          message: `No se encontró un cliente con DNI ${dni}`
        });
      }
  
      res.status(200).json({
        success: true,
        data: {
          idCliente: cliente.idCliente,
          documento: cliente.documento,
          nombre: cliente.nombre,
          apellidoPaterno: cliente.apellidoPaterno,
          apellidoMaterno: cliente.apellidoMaterno,
          correo: cliente.correo,
          telefono: cliente.telefono,
          ubicacion: {
            direccion: cliente.ubicacion.direccion,
            distrito: cliente.ubicacion.distrito.nombre,
            provincia: cliente.ubicacion.distrito.provincia.nombre,
            region: cliente.ubicacion.distrito.provincia.region.nombre
          }
        }
      });
  
    } catch (error) {
      console.error('Error al buscar cliente:', error);
      res.status(500).json({
        success: false,
        error: 'Error interno del servidor',
        message: error.message
      });
    }
  };
  

  
  // POST - Generar minería de datos
    const generarMineria = async (req, res) => {
        try {
          const { spawn } = require('child_process');
          const path = require('path');
          
          console.log('Iniciando proceso de minería de datos...');
          
          // Ruta al script de Python
          const pythonScript = path.join(__dirname, '../services/ejecutar_mineria.py');
          const DATABASE_URL = process.env.DATABASE_URL;
          
          console.log('Ejecutando script:', pythonScript);
          console.log('DATABASE_URL:', DATABASE_URL ? 'Configurada' : 'No configurada');
          
          // Ejecutar script de Python
          const pythonProcess = spawn('python', [pythonScript, DATABASE_URL]);
          
          let dataBuffer = '';
          let errorBuffer = '';
          
          pythonProcess.stdout.on('data', (data) => {
            const chunk = data.toString();
            console.log('Python stdout:', chunk);
            dataBuffer += chunk;
          });
          
          pythonProcess.stderr.on('data', (data) => {
            const chunk = data.toString();
            console.log('Python stderr:', chunk);
            errorBuffer += chunk;
          });
          
          pythonProcess.on('close', (code) => {
            console.log(`Proceso Python terminado con código: ${code}`);
            
            if (code !== 0) {
              console.error('Error en proceso Python:', errorBuffer);
              return res.status(500).json({
                success: false,
                error: 'Error al ejecutar análisis de minería',
                details: errorBuffer
              });
            }
            
            try {
              // Buscar el JSON en la salida (último JSON válido)
              const lines = dataBuffer.trim().split('\n');
              let jsonResult = null;
              
              // Buscar desde el final hacia atrás para encontrar el JSON de resultado
              for (let i = lines.length - 1; i >= 0; i--) {
                const line = lines[i].trim();
                if (line.startsWith('{') && line.endsWith('}')) {
                  try {
                    jsonResult = JSON.parse(line);
                    break;
                  } catch (e) {
                    continue;
                  }
                }
              }
              
              if (!jsonResult) {
                throw new Error('No se encontró respuesta JSON válida en la salida');
              }
              
              if (jsonResult.success) {
                res.status(200).json({
                  success: true,
                  message: 'Minería de datos generada exitosamente',
                  data: jsonResult
                });
              } else {
                res.status(500).json({
                  success: false,
                  error: jsonResult.error || 'Error desconocido en minería'
                });
              }
              
            } catch (parseError) {
              console.error('Error al parsear respuesta:', parseError);
              console.log('Salida completa:', dataBuffer);
              res.status(500).json({
                success: false,
                error: 'Error al procesar resultados de minería',
                details: parseError.message,
                rawOutput: dataBuffer
              });
            }
          });
          
          pythonProcess.on('error', (error) => {
            console.error('Error al iniciar proceso Python:', error);
            res.status(500).json({
              success: false,
              error: 'Error al iniciar proceso Python',
              details: error.message
            });
          });
          
          // Timeout de 5 minutos
          const timeout = setTimeout(() => {
            console.log('Timeout alcanzado, matando proceso...');
            pythonProcess.kill('SIGTERM');
            res.status(408).json({
              success: false,
              error: 'Timeout: El análisis de minería tomó demasiado tiempo'
            });
          }, 300000); // 5 minutos
          
          // Limpiar timeout si el proceso termina antes
          pythonProcess.on('close', () => {
            clearTimeout(timeout);
          });
          
        } catch (error) {
          console.error('Error al generar minería:', error);
          res.status(500).json({
            success: false,
            error: 'Error interno del servidor',
            message: error.message
          });
        }
      };
  
  // CAMBIAR la línea de module.exports por esta:
  module.exports = {
    crearSalida,
    obtenerSalidas,
    obtenerProductos,
    buscarClientePorDNI,
    generarMineria
  };

