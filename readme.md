```sql
CREATE TABLE `tweet` (
  `tweet_id` varchar(64) NOT NULL DEFAULT '' COMMENT 'tweet id',
  `user_id` varchar(64) NOT NULL DEFAULT '' COMMENT 'user id',
  `text` varchar(1024) NOT NULL DEFAULT '' COMMENT 'tweet 内容',
  `hash_tags` varchar(512) NOT NULL DEFAULT '' COMMENT 'hash tag 列表',
  `url` varchar(64) NOT NULL DEFAULT '' COMMENT 'url',
  `nbr_retweet` int(9) unsigned NOT NULL DEFAULT '0' COMMENT 'number of retweetk',
  `nbr_favorite` int(9) unsigned NOT NULL DEFAULT '0' COMMENT 'nuber of favorite',
  `nbr_reply` int(9) unsigned NOT NULL DEFAULT '0' COMMENT 'nuber of reply',
  `datetime` timestamp NULL COMMENT '发表时间',
  `has_media` tinyint(4) NOT NULL DEFAULT '0' COMMENT '0-false, 1-true',
  `medias` varchar(256) DEFAULT '' COMMENT 'media 列表',
  `is_reply` tinyint(4) NOT NULL DEFAULT '0' COMMENT '0-false, 1-true',
  `is_retweet` tinyint(4) NOT NULL DEFAULT '0' COMMENT '0-false, 1-true',
  PRIMARY KEY (`tweet_id`),
  KEY `idx_uid` (`user_id`),
  KEY `idx_datetime` (`datetime`),
  KEY `idx_nbr_retweet` (`nbr_retweet`),
  KEY `idx_nbr_favorite` (`nbr_favorite`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4

CREATE TABLE `user` (
  `user_id` varchar(64) NOT NULL DEFAULT '' COMMENT 'user id',
  `name` varchar(200) NOT NULL DEFAULT '' COMMENT '用户名',
  `screen_name` varchar(200) NOT NULL DEFAULT '' COMMENT '显示用户名',
  `avatar` varchar(256) NOT NULL DEFAULT '',
  PRIMARY KEY (`user_id`),
  KEY `idx_sname` (`screen_name`(191))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
```