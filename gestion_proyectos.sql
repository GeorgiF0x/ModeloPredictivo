-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Servidor: 127.0.0.1
-- Tiempo de generación: 05-12-2024 a las 01:09:13
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
CREATE DATABASE IF NOT EXISTS `gestion_proyectos` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE `gestion_proyectos`;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `certificaciones_requeridas`
--

CREATE TABLE `certificaciones_requeridas` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `descripcion` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `certificaciones_requeridas`
--

INSERT INTO `certificaciones_requeridas` (`id`, `nombre`, `descripcion`) VALUES
(1, 'Serbatic', NULL),
(2, 'VaSS', NULL),
(3, 'Otra', NULL);

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
  `nombre` varchar(255) NOT NULL,
  `descripcion` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `entregables`
--

INSERT INTO `entregables` (`id`, `nombre`, `descripcion`) VALUES
(1, 'Económico y CVS', NULL),
(2, 'Documentación Técnica', NULL),
(3, 'Otro', NULL);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `experiencia_requerida`
--

CREATE TABLE `experiencia_requerida` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `descripcion` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `experiencia_requerida`
--

INSERT INTO `experiencia_requerida` (`id`, `nombre`, `descripcion`) VALUES
(1, 'Específica', NULL),
(2, 'Generalista', NULL);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `facturacion_anual`
--

CREATE TABLE `facturacion_anual` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `descripcion` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `facturacion_anual`
--

INSERT INTO `facturacion_anual` (`id`, `nombre`, `descripcion`) VALUES
(1, 'Menos de 250k', NULL),
(2, 'Entre 250k y 1M', NULL),
(3, 'Más de 1M', NULL);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `fortaleza_tecnologica`
--

CREATE TABLE `fortaleza_tecnologica` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `nivel` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `fortaleza_tecnologica`
--

INSERT INTO `fortaleza_tecnologica` (`id`, `nombre`, `nivel`) VALUES
(1, 'Nivel Básico', NULL),
(2, 'Nivel Alto', NULL),
(3, 'Nivel Experto', NULL);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `idiomas`
--

CREATE TABLE `idiomas` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `nivel` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `idiomas`
--

INSERT INTO `idiomas` (`id`, `nombre`, `nivel`) VALUES
(1, 'Sí', NULL),
(2, 'No', NULL);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `lugar_de_trabajo`
--

CREATE TABLE `lugar_de_trabajo` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `nivel` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `lugar_de_trabajo`
--

INSERT INTO `lugar_de_trabajo` (`id`, `nombre`, `nivel`) VALUES
(1, 'Presencial', NULL),
(2, 'Remoto', NULL),
(3, 'Híbrido', NULL);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `precio_hora`
--

CREATE TABLE `precio_hora` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `nivel` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `precio_hora`
--

INSERT INTO `precio_hora` (`id`, `nombre`, `nivel`) VALUES
(1, 'Dentro del mercado', NULL),
(2, 'Por debajo del mercado', NULL),
(3, 'Por encima del mercado', NULL);

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
  `fecha_fin` datetime(6) DEFAULT NULL,
  `fecha_inicio` datetime(6) DEFAULT NULL,
  `fecha_registro` datetime(6) DEFAULT NULL,
  `nombre_proyecto` varchar(255) DEFAULT NULL,
  `presupuesto` double DEFAULT NULL,
  `resultado` tinyint(4) DEFAULT NULL CHECK (`resultado` between 0 and 1),
  `cliente` varchar(255) DEFAULT NULL,
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
  `porcentaje_exito` double DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `proyectos`
--

INSERT INTO `proyectos` (`id`, `duracion`, `fecha_fin`, `fecha_inicio`, `fecha_registro`, `nombre_proyecto`, `presupuesto`, `resultado`, `cliente`, `certificaciones_requeridas_id`, `entregables_id`, `experiencia_requerida_id`, `facturacion_anual_id`, `fortaleza_tecnologica_id`, `idiomas_id`, `lugar_trabajo_id`, `numero_perfiles_requeridos`, `precio_hora_id`, `solvencia_economica_empresa`, `titulaciones_empleados_id`, `volumetria_id`, `porcentaje_exito`) VALUES
(6, 12, '2024-12-31 01:00:00.000000', '2024-01-01 01:00:00.000000', '2024-12-04 13:00:00.000000', 'Proyecto Completo', 150000, 1, 'Cliente Ejemplo', 1, 2, 1, 1, 1, 1, 1, 5, 1, 'Alta', 1, 1, 85.5);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `proyectos_tecnologias`
--

CREATE TABLE `proyectos_tecnologias` (
  `id_tecnologia` bigint(20) NOT NULL,
  `id_proyecto` bigint(20) NOT NULL,
  `proyecto_id` bigint(20) NOT NULL,
  `tecnologia_id` bigint(20) NOT NULL
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
  `nombre` varchar(255) NOT NULL,
  `nivel` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `titulacion_empleados`
--

INSERT INTO `titulacion_empleados` (`id`, `nombre`, `nivel`) VALUES
(1, 'Titulación Universitaria', NULL),
(2, 'Titulación FP', NULL),
(3, 'Sin Titulación', NULL);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `volumetria`
--

CREATE TABLE `volumetria` (
  `id` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `nivel` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `volumetria`
--

INSERT INTO `volumetria` (`id`, `nombre`, `nivel`) VALUES
(1, 'Sí', NULL),
(2, 'No', NULL);

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
  ADD KEY `FKo19mesjeuwaciqplg2fsqdn07` (`certificaciones_requeridas_id`),
  ADD KEY `FKa6kcwxwg6d8a9acxm40p602ti` (`entregables_id`),
  ADD KEY `FK30bhww1yuhi1ecn0w096ofec2` (`experiencia_requerida_id`),
  ADD KEY `FKi9e1ivys9ip84jqyuldv2h7ns` (`facturacion_anual_id`),
  ADD KEY `FKlqhsj5bckgaba5kricknvj13l` (`fortaleza_tecnologica_id`),
  ADD KEY `FKgsge4t7fg7gvnnucy92dbdhtt` (`idiomas_id`),
  ADD KEY `FKhf95jdmjlsl71yrdvc1e0ug7o` (`lugar_trabajo_id`),
  ADD KEY `FKqk6xls5v1f8ybibvbwwimdeur` (`precio_hora_id`),
  ADD KEY `FK54uxg0wfdlsfe3va04y7bhhyk` (`titulaciones_empleados_id`),
  ADD KEY `FKbnwm6jvi1ligb5be70fdsevon` (`volumetria_id`);

--
-- Indices de la tabla `proyectos_tecnologias`
--
ALTER TABLE `proyectos_tecnologias`
  ADD KEY `FKb2gtdar522ce8uu03xkqs236n` (`id_proyecto`),
  ADD KEY `FKk881gugdfv9dmc0sv7eij9gk2` (`id_tecnologia`),
  ADD KEY `FKas9xs6flsvrfsfviavec95872` (`tecnologia_id`),
  ADD KEY `FKbdy0sen1ai19mq2d2tbt5nyp7` (`proyecto_id`);

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
-- AUTO_INCREMENT de la tabla `proyectos`
--
ALTER TABLE `proyectos`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;

--
-- Restricciones para tablas volcadas
--

--
-- Filtros para la tabla `proyectos`
--
ALTER TABLE `proyectos`
  ADD CONSTRAINT `FK30bhww1yuhi1ecn0w096ofec2` FOREIGN KEY (`experiencia_requerida_id`) REFERENCES `experiencia_requerida` (`id`),
  ADD CONSTRAINT `FK54uxg0wfdlsfe3va04y7bhhyk` FOREIGN KEY (`titulaciones_empleados_id`) REFERENCES `titulacion_empleados` (`id`),
  ADD CONSTRAINT `FKa6kcwxwg6d8a9acxm40p602ti` FOREIGN KEY (`entregables_id`) REFERENCES `entregables` (`id`),
  ADD CONSTRAINT `FKbnwm6jvi1ligb5be70fdsevon` FOREIGN KEY (`volumetria_id`) REFERENCES `volumetria` (`id`),
  ADD CONSTRAINT `FKgsge4t7fg7gvnnucy92dbdhtt` FOREIGN KEY (`idiomas_id`) REFERENCES `idiomas` (`id`),
  ADD CONSTRAINT `FKhf95jdmjlsl71yrdvc1e0ug7o` FOREIGN KEY (`lugar_trabajo_id`) REFERENCES `lugar_de_trabajo` (`id`),
  ADD CONSTRAINT `FKi9e1ivys9ip84jqyuldv2h7ns` FOREIGN KEY (`facturacion_anual_id`) REFERENCES `facturacion_anual` (`id`),
  ADD CONSTRAINT `FKlqhsj5bckgaba5kricknvj13l` FOREIGN KEY (`fortaleza_tecnologica_id`) REFERENCES `fortaleza_tecnologica` (`id`),
  ADD CONSTRAINT `FKo19mesjeuwaciqplg2fsqdn07` FOREIGN KEY (`certificaciones_requeridas_id`) REFERENCES `certificaciones_requeridas` (`id`),
  ADD CONSTRAINT `FKqk6xls5v1f8ybibvbwwimdeur` FOREIGN KEY (`precio_hora_id`) REFERENCES `precio_hora` (`id`);

--
-- Filtros para la tabla `proyectos_tecnologias`
--
ALTER TABLE `proyectos_tecnologias`
  ADD CONSTRAINT `FKas9xs6flsvrfsfviavec95872` FOREIGN KEY (`tecnologia_id`) REFERENCES `tecnologias` (`id`),
  ADD CONSTRAINT `FKbdy0sen1ai19mq2d2tbt5nyp7` FOREIGN KEY (`proyecto_id`) REFERENCES `proyectos` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
