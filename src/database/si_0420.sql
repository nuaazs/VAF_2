SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for call_records
-- ----------------------------
DROP TABLE IF EXISTS `call_records`;
CREATE TABLE `call_records`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_1
-- ----------------------------
DROP TABLE IF EXISTS `call_records_1`;
CREATE TABLE `call_records_1`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_10
-- ----------------------------
DROP TABLE IF EXISTS `call_records_10`;
CREATE TABLE `call_records_10`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_11
-- ----------------------------
DROP TABLE IF EXISTS `call_records_11`;
CREATE TABLE `call_records_11`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_12
-- ----------------------------
DROP TABLE IF EXISTS `call_records_12`;
CREATE TABLE `call_records_12`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 17 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_2
-- ----------------------------
DROP TABLE IF EXISTS `call_records_2`;
CREATE TABLE `call_records_2`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_3
-- ----------------------------
DROP TABLE IF EXISTS `call_records_3`;
CREATE TABLE `call_records_3`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_4
-- ----------------------------
DROP TABLE IF EXISTS `call_records_4`;
CREATE TABLE `call_records_4`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_5
-- ----------------------------
DROP TABLE IF EXISTS `call_records_5`;
CREATE TABLE `call_records_5`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_6
-- ----------------------------
DROP TABLE IF EXISTS `call_records_6`;
CREATE TABLE `call_records_6`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_7
-- ----------------------------
DROP TABLE IF EXISTS `call_records_7`;
CREATE TABLE `call_records_7`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_8
-- ----------------------------
DROP TABLE IF EXISTS `call_records_8`;
CREATE TABLE `call_records_8`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for call_records_9
-- ----------------------------
DROP TABLE IF EXISTS `call_records_9`;
CREATE TABLE `call_records_9`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '通话记录ID',
  `calling_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '主叫号码',
  `called_number` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫号码',
  `tx_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '腾讯研判类型',
  `kdxf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '讯飞研判类型',
  `xf_casetype` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '欣方研判类型',
  `hit_keyword` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关键字',
  `content_text` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'ASR内容',
  `calling_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `calling_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `called_talk_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫漫游地',
  `called_belong_city` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被叫归属地',
  `talk_start_time` datetime NULL DEFAULT NULL,
  `talk_end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT NULL,
  `is_download` tinyint(4) NULL DEFAULT 0,
  `is_extract` tinyint(4) NULL DEFAULT 0 COMMENT '是否提取过声纹',
  `is_black` tinyint(255) NULL DEFAULT NULL COMMENT '是否判黑 1 为判黑  ',
  `is single _voice` tinyint(4) NULL DEFAULT NULL COMMENT '是否未单人声',
  `source` tinyint(4) NULL DEFAULT NULL COMMENT '境内1境外0',
  `data_source` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `g4mark` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `call_records_11_record_id_uindex`(`record_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for check_list
-- ----------------------------
DROP TABLE IF EXISTS `check_list`;
CREATE TABLE `check_list`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` varchar(11) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '通话记录ID',
  `wavs_list` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '音频文件地址 多个用逗号分割',
  `call_record_wav` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '通话音频',
  `record_id_list` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `is_black_01` tinyint(4) NULL DEFAULT NULL COMMENT '第一次审核是否是诈骗   1为 判黑2 为撤回 ',
  `is_black_02` tinyint(4) NULL DEFAULT NULL COMMENT '第二次审核是否为诈骗  1为 判黑 2 为撤回 ',
  `is_black_01_name` varchar(16) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '第一次审核人姓名',
  `is_black_02_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '第二次审核人姓名',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `source` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 7 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '审核表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for hit
-- ----------------------------
DROP TABLE IF EXISTS `hit`;
CREATE TABLE `hit`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `show_phone` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '用于展示的手机号',
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `phone_type` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `area_code` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `self_test_score_mean` float NULL DEFAULT NULL,
  `self_test_score_min` float NULL DEFAULT NULL,
  `self_test_score_max` float NULL DEFAULT NULL,
  `call_begintime` datetime NULL DEFAULT NULL,
  `call_endtime` datetime NULL DEFAULT NULL,
  `class_number` int(11) NULL DEFAULT NULL,
  `hit_time` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  `blackbase_phone` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `blackbase_id` varchar(12) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '黑库命中id',
  `hit_status` int(11) NULL DEFAULT NULL,
  `hit_score` varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `top_10` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `valid_length` int(11) NULL DEFAULT 0,
  `is_grey` int(11) NULL DEFAULT 0,
  `content_text` varchar(2550) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT 'asr结果',
  `hit_keyword` varchar(2550) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '是否命中关键字',
  `keyword` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '命中的关键字',
  `final_score` float NULL DEFAULT NULL COMMENT '最终得分',
  PRIMARY KEY (`id`, `phone`, `file_url`) USING BTREE,
  UNIQUE INDEX `file_url`(`file_url`) USING BTREE,
  UNIQUE INDEX `phone`(`phone`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1281 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log
-- ----------------------------
DROP TABLE IF EXISTS `log`;
CREATE TABLE `log`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `valid_length` int(11) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_1
-- ----------------------------
DROP TABLE IF EXISTS `log_1`;
CREATE TABLE `log_1`  (
  `id` int(11) NULL DEFAULT NULL,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for log_10
-- ----------------------------
DROP TABLE IF EXISTS `log_10`;
CREATE TABLE `log_10`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 20 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_11
-- ----------------------------
DROP TABLE IF EXISTS `log_11`;
CREATE TABLE `log_11`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_12
-- ----------------------------
DROP TABLE IF EXISTS `log_12`;
CREATE TABLE `log_12`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 6 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_13
-- ----------------------------
DROP TABLE IF EXISTS `log_13`;
CREATE TABLE `log_13`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 40 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_14
-- ----------------------------
DROP TABLE IF EXISTS `log_14`;
CREATE TABLE `log_14`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 195 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_15
-- ----------------------------
DROP TABLE IF EXISTS `log_15`;
CREATE TABLE `log_15`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_16
-- ----------------------------
DROP TABLE IF EXISTS `log_16`;
CREATE TABLE `log_16`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 134 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_17
-- ----------------------------
DROP TABLE IF EXISTS `log_17`;
CREATE TABLE `log_17`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_18
-- ----------------------------
DROP TABLE IF EXISTS `log_18`;
CREATE TABLE `log_18`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_19
-- ----------------------------
DROP TABLE IF EXISTS `log_19`;
CREATE TABLE `log_19`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_2
-- ----------------------------
DROP TABLE IF EXISTS `log_2`;
CREATE TABLE `log_2`  (
  `id` int(11) NULL DEFAULT NULL,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for log_20
-- ----------------------------
DROP TABLE IF EXISTS `log_20`;
CREATE TABLE `log_20`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 185 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_21
-- ----------------------------
DROP TABLE IF EXISTS `log_21`;
CREATE TABLE `log_21`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_22
-- ----------------------------
DROP TABLE IF EXISTS `log_22`;
CREATE TABLE `log_22`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_23
-- ----------------------------
DROP TABLE IF EXISTS `log_23`;
CREATE TABLE `log_23`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_24
-- ----------------------------
DROP TABLE IF EXISTS `log_24`;
CREATE TABLE `log_24`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1058 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_25
-- ----------------------------
DROP TABLE IF EXISTS `log_25`;
CREATE TABLE `log_25`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_26
-- ----------------------------
DROP TABLE IF EXISTS `log_26`;
CREATE TABLE `log_26`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_27
-- ----------------------------
DROP TABLE IF EXISTS `log_27`;
CREATE TABLE `log_27`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 80 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_28
-- ----------------------------
DROP TABLE IF EXISTS `log_28`;
CREATE TABLE `log_28`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 16 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_29
-- ----------------------------
DROP TABLE IF EXISTS `log_29`;
CREATE TABLE `log_29`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_3
-- ----------------------------
DROP TABLE IF EXISTS `log_3`;
CREATE TABLE `log_3`  (
  `id` int(11) NULL DEFAULT NULL,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for log_30
-- ----------------------------
DROP TABLE IF EXISTS `log_30`;
CREATE TABLE `log_30`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_31
-- ----------------------------
DROP TABLE IF EXISTS `log_31`;
CREATE TABLE `log_31`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 742 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_4
-- ----------------------------
DROP TABLE IF EXISTS `log_4`;
CREATE TABLE `log_4`  (
  `id` int(11) NULL DEFAULT NULL,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for log_5
-- ----------------------------
DROP TABLE IF EXISTS `log_5`;
CREATE TABLE `log_5`  (
  `id` int(11) NULL DEFAULT NULL,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for log_6
-- ----------------------------
DROP TABLE IF EXISTS `log_6`;
CREATE TABLE `log_6`  (
  `id` int(11) NULL DEFAULT NULL,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for log_7
-- ----------------------------
DROP TABLE IF EXISTS `log_7`;
CREATE TABLE `log_7`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 132 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_8
-- ----------------------------
DROP TABLE IF EXISTS `log_8`;
CREATE TABLE `log_8`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for log_9
-- ----------------------------
DROP TABLE IF EXISTS `log_9`;
CREATE TABLE `log_9`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `show_phone` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `time` datetime NULL DEFAULT NULL,
  `action_type` int(11) NULL DEFAULT NULL,
  `err_type` int(11) NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `preprocessed_file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `message` varchar(1280) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `before_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `after_length` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for speaker
-- ----------------------------
DROP TABLE IF EXISTS `speaker`;
CREATE TABLE `speaker`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `file_url` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `phone` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `show_phone` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '用于展示的手机号',
  `register_time` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  `phone_type` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `area_code` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `self_test_score_mean` float NULL DEFAULT NULL,
  `self_test_score_min` float NULL DEFAULT NULL,
  `self_test_score_max` float NULL DEFAULT NULL,
  `call_begintime` datetime NULL DEFAULT NULL,
  `call_endtime` datetime NULL DEFAULT NULL,
  `delete_time` datetime NULL DEFAULT NULL,
  `status` int(11) NULL DEFAULT 1 COMMENT '状态改为0',
  `class_number` int(11) NULL DEFAULT 999,
  `hit_count` int(11) NULL DEFAULT 0,
  `input_reason` int(11) NULL DEFAULT 1,
  `delete_reason` int(11) NULL DEFAULT 0,
  `delete_remark` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '0',
  `valid_length` int(11) NULL DEFAULT 0,
  `preprocessed_file_url` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`, `phone`) USING BTREE,
  UNIQUE INDEX `phone`(`phone`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for t_key_word
-- ----------------------------
DROP TABLE IF EXISTS `t_key_word`;
CREATE TABLE `t_key_word`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `key_word` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `value` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 15 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for t_relevance_record
-- ----------------------------
DROP TABLE IF EXISTS `t_relevance_record`;
CREATE TABLE `t_relevance_record`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `record_id` int(11) NULL DEFAULT NULL COMMENT '待审核的任务id',
  `relevance_record_id` int(11) NULL DEFAULT NULL COMMENT '关联的任务id ',
  `relevance_wav` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关联的音频地址',
  `check_list_id` int(11) NULL DEFAULT NULL COMMENT '审核表的id',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 9 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for t_role
-- ----------------------------
DROP TABLE IF EXISTS `t_role`;
CREATE TABLE `t_role`  (
  `id` int(11) NOT NULL,
  `pid` int(11) NULL DEFAULT NULL,
  `role_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for t_user
-- ----------------------------
DROP TABLE IF EXISTS `t_user`;
CREATE TABLE `t_user`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `real_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `password` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `phone` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT NULL,
  `role_id` int(11) NULL DEFAULT NULL,
  `delete_flag` int(11) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 66 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for t_white_phone
-- ----------------------------
DROP TABLE IF EXISTS `t_white_phone`;
CREATE TABLE `t_white_phone`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `phone` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '手机号',
  `create_user` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '创建人',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_phone`(`phone`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 182 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;