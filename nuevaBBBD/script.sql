-- Crear la base de datos
CREATE DATABASE IF NOT EXISTS gestion_proyectos;
USE gestion_proyectos;

-- Crear la tabla principal
CREATE TABLE proyectos (
    id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    duracion INT(11) NOT NULL,
    fecha_fin DATE DEFAULT NULL,
    fecha_inicio DATE DEFAULT NULL,
    fecha_registro DATETIME(6) DEFAULT NULL,
    nombre_proyecto VARCHAR(255) DEFAULT NULL,
    presupuesto DOUBLE DEFAULT NULL,
    recursos_asignados VARBINARY(255) DEFAULT NULL,
    resultado TINYINT(4) DEFAULT NULL CHECK (resultado BETWEEN 0 AND 1),
    cliente_id BIGINT(20) DEFAULT NULL,
    certificaciones_requeridas_id INT DEFAULT NULL,
    entregables_id INT DEFAULT NULL,
    experiencia_requerida_id INT DEFAULT NULL,
    facturacion_anual_id INT DEFAULT NULL,
    fortaleza_tecnologica_id INT DEFAULT NULL,
    idiomas_id INT DEFAULT NULL,
    lugar_trabajo_id INT DEFAULT NULL,
    numero_perfiles_requeridos INT(11) NOT NULL,
    precio_hora_id INT DEFAULT NULL,
    solvencia_economica_empresa VARCHAR(255) DEFAULT NULL,
    titulaciones_empleados_id INT DEFAULT NULL,
    volumetria_id INT DEFAULT NULL,

    -- Claves foráneas
    FOREIGN KEY (certificaciones_requeridas_id) REFERENCES certificaciones_requeridas(id),
    FOREIGN KEY (entregables_id) REFERENCES entregables(id),
    FOREIGN KEY (experiencia_requerida_id) REFERENCES experiencia_requerida(id),
    FOREIGN KEY (facturacion_anual_id) REFERENCES facturacion_anual(id),
    FOREIGN KEY (fortaleza_tecnologica_id) REFERENCES fortaleza_tecnologica(id),
    FOREIGN KEY (idiomas_id) REFERENCES idiomas(id),
    FOREIGN KEY (lugar_trabajo_id) REFERENCES lugar_de_trabajo(id),
    FOREIGN KEY (precio_hora_id) REFERENCES precio_hora(id),
    FOREIGN KEY (titulaciones_empleados_id) REFERENCES titulacion_empleados(id),
    FOREIGN KEY (volumetria_id) REFERENCES volumetria(id)
);

-- Crear tablas de referencia para columnas categóricas
CREATE TABLE certificaciones_requeridas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);

CREATE TABLE entregables (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);

CREATE TABLE experiencia_requerida (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);

CREATE TABLE facturacion_anual (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);

CREATE TABLE fortaleza_tecnologica (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);

CREATE TABLE idiomas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);

CREATE TABLE lugar_de_trabajo (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);

CREATE TABLE precio_hora (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);

CREATE TABLE titulacion_empleados (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);

CREATE TABLE volumetria (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL
);


--insertar opciones 
-- Insertar opciones en certificaciones_requeridas
INSERT INTO certificaciones_requeridas (nombre) VALUES
('Serbatic'),
('VaSS'),
('Otra');

-- Insertar opciones en entregables
INSERT INTO entregables (nombre) VALUES
('Económico y CVS'),
('Documentación Técnica'),
('Otro');

-- Insertar opciones en experiencia_requerida
INSERT INTO experiencia_requerida (nombre) VALUES
('Específica'),
('Generalista');

-- Insertar opciones en facturacion_anual
INSERT INTO facturacion_anual (nombre) VALUES
('Menos de 250k'),
('Entre 250k y 1M'),
('Más de 1M');

-- Insertar opciones en fortaleza_tecnologica
INSERT INTO fortaleza_tecnologica (nombre) VALUES
('Nivel Básico'),
('Nivel Alto'),
('Nivel Experto');

-- Insertar opciones en idiomas
INSERT INTO idiomas (nombre) VALUES
('Sí'),
('No');

-- Insertar opciones en lugar_de_trabajo
INSERT INTO lugar_de_trabajo (nombre) VALUES
('Presencial'),
('Remoto'),
('Híbrido');

-- Insertar opciones en precio_hora
INSERT INTO precio_hora (nombre) VALUES
('Dentro del mercado'),
('Por debajo del mercado'),
('Por encima del mercado');

-- Insertar opciones en titulacion_empleados
INSERT INTO titulacion_empleados (nombre) VALUES
('Titulación Universitaria'),
('Titulación FP'),
('Sin Titulación');

-- Insertar opciones en volumetria
INSERT INTO volumetria (nombre) VALUES
('Sí'),
('No');
 