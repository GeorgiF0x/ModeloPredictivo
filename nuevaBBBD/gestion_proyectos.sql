-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Servidor: 127.0.0.1
-- Tiempo de generación: 03-12-2024 a las 10:01:26
-- Versión del servidor: 10.4.32-MariaDB
-- Versión de PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `gestion_proyectos`
--

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `certificaciones_requeridas`
--

CREATE TABLE `certificaciones_requeridas` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `certificaciones_requeridas`
--

INSERT INTO `certificaciones_requeridas` (`id`, `nombre`) VALUES
(1, 'Serbatic'),
(2, 'VaSS'),
(3, 'Otra');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `clientes`
--

CREATE TABLE `clientes` (
  `id` bigint(20) NOT NULL,
  `nombre` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `entregables`
--

CREATE TABLE `entregables` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `entregables`
--

INSERT INTO `entregables` (`id`, `nombre`) VALUES
(1, 'Económico y CVS'),
(2, 'Documentación Técnica'),
(3, 'Otro');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `experiencia_requerida`
--

CREATE TABLE `experiencia_requerida` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `experiencia_requerida`
--

INSERT INTO `experiencia_requerida` (`id`, `nombre`) VALUES
(1, 'Específica'),
(2, 'Generalista');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `facturacion_anual`
--

CREATE TABLE `facturacion_anual` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `facturacion_anual`
--

INSERT INTO `facturacion_anual` (`id`, `nombre`) VALUES
(1, 'Menos de 250k'),
(2, 'Entre 250k y 1M'),
(3, 'Más de 1M');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `fortaleza_tecnologica`
--

CREATE TABLE `fortaleza_tecnologica` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `fortaleza_tecnologica`
--

INSERT INTO `fortaleza_tecnologica` (`id`, `nombre`) VALUES
(1, 'Nivel Básico'),
(2, 'Nivel Alto'),
(3, 'Nivel Experto');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `idiomas`
--

CREATE TABLE `idiomas` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `idiomas`
--

INSERT INTO `idiomas` (`id`, `nombre`) VALUES
(1, 'Sí'),
(2, 'No');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `lugar_de_trabajo`
--

CREATE TABLE `lugar_de_trabajo` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `lugar_de_trabajo`
--

INSERT INTO `lugar_de_trabajo` (`id`, `nombre`) VALUES
(1, 'Presencial'),
(2, 'Remoto'),
(3, 'Híbrido');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `precio_hora`
--

CREATE TABLE `precio_hora` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `precio_hora`
--

INSERT INTO `precio_hora` (`id`, `nombre`) VALUES
(1, 'Dentro del mercado'),
(2, 'Por debajo del mercado'),
(3, 'Por encima del mercado');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `predicciones`
--

CREATE TABLE `predicciones` (
  `id` bigint(20) NOT NULL,
  `fecha_prediccion` timestamp NULL DEFAULT current_timestamp(),
  `probabilidad_exito` double DEFAULT NULL,
  `proyecto_id` bigint(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `proyectos`
--

CREATE TABLE `proyectos` (
  `id` bigint(20) NOT NULL,
  `duracion` int(11) NOT NULL,
  `fecha_fin` date DEFAULT NULL,
  `fecha_inicio` date DEFAULT NULL,
  `fecha_registro` datetime(6) DEFAULT NULL,
  `nombre_proyecto` varchar(255) DEFAULT NULL,
  `presupuesto` double DEFAULT NULL,
  `recursos_asignados` varbinary(255) DEFAULT NULL,
  `cliente_id` bigint(20) DEFAULT NULL,
  `certificaciones_requeridas_id` int(11) DEFAULT NULL,
  `entregables_id` int(11) DEFAULT NULL,
  `experiencia_requerida_id` int(11) DEFAULT NULL,
  `facturacion_anual_id` int(11) DEFAULT NULL,
  `fortaleza_tecnologica_id` int(11) DEFAULT NULL,
  `idiomas_id` int(11) DEFAULT NULL,
  `lugar_trabajo_id` int(11) DEFAULT NULL,
  `numero_perfiles_requeridos` int(11) NOT NULL,
  `precio_hora_id` int(11) DEFAULT NULL,
  `solvencia_economica_empresa` varchar(255) DEFAULT NULL,
  `titulaciones_empleados_id` int(11) DEFAULT NULL,
  `volumetria_id` int(11) DEFAULT NULL,
  `porcentaje_exito` double DEFAULT NULL,
  `entregable_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `proyectos`
--

INSERT INTO `proyectos` (`id`, `duracion`, `fecha_fin`, `fecha_inicio`, `fecha_registro`, `nombre_proyecto`, `presupuesto`, `recursos_asignados`, `cliente_id`, `certificaciones_requeridas_id`, `entregables_id`, `experiencia_requerida_id`, `facturacion_anual_id`, `fortaleza_tecnologica_id`, `idiomas_id`, `lugar_trabajo_id`, `numero_perfiles_requeridos`, `precio_hora_id`, `solvencia_economica_empresa`, `titulaciones_empleados_id`, `volumetria_id`, `porcentaje_exito`, `entregable_id`) VALUES
(1, 12, '2024-12-31', '2024-01-01', '2024-12-02 13:31:24.000000', 'Proyecto de Ejemplo', 100000, NULL, NULL, 1, 2, 1, 2, 3, 1, 2, 3, 1, 'Solvencia económica empresa A', 2, 1, 80.5, 1);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `proyectos_tecnologias`
--

CREATE TABLE `proyectos_tecnologias` (
  `id_tecnologia` bigint(20) NOT NULL,
  `id_proyecto` bigint(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `tecnologias`
--

CREATE TABLE `tecnologias` (
  `id` bigint(20) NOT NULL,
  `frecuencia_uso` bigint(20) DEFAULT NULL,
  `nombre` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `tecnologias`
--

INSERT INTO `tecnologias` (`id`, `frecuencia_uso`, `nombre`) VALUES
(1, 100, 'Java'),
(2, 150, 'Python'),
(3, 200, 'JavaScript'),
(4, 50, 'C#'),
(5, 30, 'Ruby');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `titulacion_empleados`
--

CREATE TABLE `titulacion_empleados` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `titulacion_empleados`
--

INSERT INTO `titulacion_empleados` (`id`, `nombre`) VALUES
(1, 'Titulación Universitaria'),
(2, 'Titulación FP'),
(3, 'Sin Titulación');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `volumetria`
--

CREATE TABLE `volumetria` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `volumetria`
--

INSERT INTO `volumetria` (`id`, `nombre`) VALUES
(1, 'Sí'),
(2, 'No');

--
-- Índices para tablas volcadas
--

--
-- Indices de la tabla `certificaciones_requeridas`
--
ALTER TABLE `certificaciones_requeridas`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `clientes`
--
ALTER TABLE `clientes`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `entregables`
--
ALTER TABLE `entregables`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `experiencia_requerida`
--
ALTER TABLE `experiencia_requerida`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `facturacion_anual`
--
ALTER TABLE `facturacion_anual`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `fortaleza_tecnologica`
--
ALTER TABLE `fortaleza_tecnologica`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `idiomas`
--
ALTER TABLE `idiomas`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `lugar_de_trabajo`
--
ALTER TABLE `lugar_de_trabajo`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `precio_hora`
--
ALTER TABLE `precio_hora`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `predicciones`
--
ALTER TABLE `predicciones`
  ADD PRIMARY KEY (`id`),
  ADD KEY `FK7jfrf74tfe2d1vjcyjfu3ftus` (`proyecto_id`);

--
-- Indices de la tabla `proyectos`
--
ALTER TABLE `proyectos`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `UKspw880kbwcnhsj86ogs1b04in` (`entregable_id`),
  ADD KEY `certificaciones_requeridas_id` (`certificaciones_requeridas_id`),
  ADD KEY `entregables_id` (`entregables_id`),
  ADD KEY `experiencia_requerida_id` (`experiencia_requerida_id`),
  ADD KEY `facturacion_anual_id` (`facturacion_anual_id`),
  ADD KEY `fortaleza_tecnologica_id` (`fortaleza_tecnologica_id`),
  ADD KEY `idiomas_id` (`idiomas_id`),
  ADD KEY `lugar_trabajo_id` (`lugar_trabajo_id`),
  ADD KEY `precio_hora_id` (`precio_hora_id`),
  ADD KEY `titulaciones_empleados_id` (`titulaciones_empleados_id`),
  ADD KEY `volumetria_id` (`volumetria_id`),
  ADD KEY `FK6ox9d0jf9jy52ha3txq9cylw9` (`cliente_id`);

--
-- Indices de la tabla `proyectos_tecnologias`
--
ALTER TABLE `proyectos_tecnologias`
  ADD KEY `FKb2gtdar522ce8uu03xkqs236n` (`id_proyecto`),
  ADD KEY `FKk881gugdfv9dmc0sv7eij9gk2` (`id_tecnologia`);

--
-- Indices de la tabla `tecnologias`
--
ALTER TABLE `tecnologias`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `titulacion_empleados`
--
ALTER TABLE `titulacion_empleados`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `volumetria`
--
ALTER TABLE `volumetria`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT de las tablas volcadas
--

--
-- AUTO_INCREMENT de la tabla `certificaciones_requeridas`
--
ALTER TABLE `certificaciones_requeridas`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT de la tabla `clientes`
--
ALTER TABLE `clientes`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `entregables`
--
ALTER TABLE `entregables`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT de la tabla `experiencia_requerida`
--
ALTER TABLE `experiencia_requerida`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `facturacion_anual`
--
ALTER TABLE `facturacion_anual`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT de la tabla `fortaleza_tecnologica`
--
ALTER TABLE `fortaleza_tecnologica`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT de la tabla `idiomas`
--
ALTER TABLE `idiomas`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `lugar_de_trabajo`
--
ALTER TABLE `lugar_de_trabajo`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT de la tabla `precio_hora`
--
ALTER TABLE `precio_hora`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT de la tabla `predicciones`
--
ALTER TABLE `predicciones`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `proyectos`
--
ALTER TABLE `proyectos`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT de la tabla `tecnologias`
--
ALTER TABLE `tecnologias`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT de la tabla `titulacion_empleados`
--
ALTER TABLE `titulacion_empleados`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT de la tabla `volumetria`
--
ALTER TABLE `volumetria`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- Restricciones para tablas volcadas
--

--
-- Filtros para la tabla `predicciones`
--
ALTER TABLE `predicciones`
  ADD CONSTRAINT `FK7jfrf74tfe2d1vjcyjfu3ftus` FOREIGN KEY (`proyecto_id`) REFERENCES `proyectos` (`id`);

--
-- Filtros para la tabla `proyectos`
--
ALTER TABLE `proyectos`
  ADD CONSTRAINT `FK6ox9d0jf9jy52ha3txq9cylw9` FOREIGN KEY (`cliente_id`) REFERENCES `clientes` (`id`),
  ADD CONSTRAINT `FKa4ulwymr5v99vmmbtpdag3q6x` FOREIGN KEY (`entregable_id`) REFERENCES `entregables` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_1` FOREIGN KEY (`certificaciones_requeridas_id`) REFERENCES `certificaciones_requeridas` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_10` FOREIGN KEY (`volumetria_id`) REFERENCES `volumetria` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_2` FOREIGN KEY (`entregables_id`) REFERENCES `entregables` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_3` FOREIGN KEY (`experiencia_requerida_id`) REFERENCES `experiencia_requerida` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_4` FOREIGN KEY (`facturacion_anual_id`) REFERENCES `facturacion_anual` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_5` FOREIGN KEY (`fortaleza_tecnologica_id`) REFERENCES `fortaleza_tecnologica` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_6` FOREIGN KEY (`idiomas_id`) REFERENCES `idiomas` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_7` FOREIGN KEY (`lugar_trabajo_id`) REFERENCES `lugar_de_trabajo` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_8` FOREIGN KEY (`precio_hora_id`) REFERENCES `precio_hora` (`id`),
  ADD CONSTRAINT `proyectos_ibfk_9` FOREIGN KEY (`titulaciones_empleados_id`) REFERENCES `titulacion_empleados` (`id`);

--
-- Filtros para la tabla `proyectos_tecnologias`
--
ALTER TABLE `proyectos_tecnologias`
  ADD CONSTRAINT `FKb2gtdar522ce8uu03xkqs236n` FOREIGN KEY (`id_proyecto`) REFERENCES `proyectos` (`id`),
  ADD CONSTRAINT `FKk881gugdfv9dmc0sv7eij9gk2` FOREIGN KEY (`id_tecnologia`) REFERENCES `tecnologias` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
