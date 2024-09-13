#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import fire
import signal
import hashlib
import urllib3

from robot.Updater import Updater
from robot.Conversation import Conversation
from robot.LifeCycleHandler import LifeCycleHandler

from robot import config, utils, constants, logging, detector

from server import server
from tools import make_json, solr_tools

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class DossBot(object):

    _profiling = False
    _debug = False

    def init(self):
        self.detector = None
        self.porcupine = None
        self.gui = None
        self._interrupted = False
        print(
            """
********************************************************
*          doss-robot - DOSS语音对话机器人           *
*               (c) 2024 SDARI             *
*               当前版本号: V0.1.0                      *
********************************************************

            后台管理端：http://{}:{}
            如需退出，可以按 Ctrl-4 组合键

""".format(
                utils.get_file_content(
                    os.path.join(constants.APP_PATH, "VERSION"), "r"
                ).strip(),
                config.get("/server/host", "0.0.0.0"),
                config.get("/server/port", "5001"),
            )
        )

        self.conversation = Conversation(self._profiling)
        self.conversation.say(
            "您好！我是上船院创新中心开发的智能机器人DOSS。欢迎和我聊天哦~", True
        )
        self.lifeCycleHandler = LifeCycleHandler(self.conversation)
        self.lifeCycleHandler.onInit()

    def _signal_handler(self, signal, frame):
        self._interrupted = True
        utils.clean()
        self.lifeCycleHandler.onKilled()

    def _detected_callback(self, is_snowboy=True):
        def _start_record():
            logger.info("开始录音")
            self.conversation.isRecording = True
            utils.setRecordable(True)

        if not utils.is_proper_time():
            logger.warning("勿扰模式开启中")
            return
        if self.conversation.isRecording:
            logger.warning("正在录音中，跳过")
            return
        if is_snowboy:
            self.conversation.interrupt()
            utils.setRecordable(False)
        self.lifeCycleHandler.onWakeup()
        if is_snowboy:
            _start_record()

    def _interrupt_callback(self):
        return self._interrupted

    def run(self):
        self.init()
        # capture SIGINT signal, e.g., Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        # 后台管理端
        server.run(self.conversation, self, debug=self._debug)
        try:
            # 初始化离线唤醒
            detector.initDetector(self)
        except AttributeError:
            logger.error("初始化离线唤醒功能失败", stack_info=True)
            pass

    def help(self):
        print(
            """=====================================================================================
    python3 doss_bot.py [命令]
    可选命令：
      md5                      - 用于计算字符串的 md5 值，常用于密码设置
      update                   - 手动更新 doss-robot
      upload [thredNum]        - 手动上传 QA 集语料，重建 solr 索引。
                                 threadNum 表示上传时开启的线程数（可选。默认值为 10）
      profiling                - 运行过程中打印耗时数据
====================================================================================="""
        )

    def md5(self, password):
        """
        计算字符串的 md5 值
        """
        return hashlib.md5(str(password).encode("utf-8")).hexdigest()

    def update(self):
        """
        更新 doss-robot
        """
        updater = Updater()
        return updater.update()

    def fetch(self):
        """
        检测 doss-robot 的更新
        """
        updater = Updater()
        updater.fetch()

    def upload(self, threadNum=10):
        """
        手动上传 QA 集语料，重建 solr 索引
        """
        try:
            qaJson = os.path.join(constants.TEMP_PATH, "qa_json")
            make_json.run(constants.getQAPath(), qaJson)
            solr_tools.clear_documents(
                config.get("/anyq/host", "0.0.0.0"),
                "collection1",
                config.get("/anyq/solr_port", "8900"),
            )
            solr_tools.upload_documents(
                config.get("/anyq/host", "0.0.0.0"),
                "collection1",
                config.get("/anyq/solr_port", "8900"),
                qaJson,
                threadNum,
            )
        except Exception as e:
            logger.error(f"上传失败：{e}", stack_info=True)

    def restart(self):
        """
        重启 doss-robot
        """
        logger.critical("程序重启...")
        try:
            self.detector.terminate()
        except AttributeError:
            pass
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def profiling(self):
        """
        运行过程中打印耗时数据
        """
        logger.info("性能调优")
        self._profiling = True
        self.run()

    def debug(self):
        """
        调试模式启动服务
        """
        logger.info("进入调试模式")
        self._debug = True
        self.run()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        bot = DossBot()
        bot.run()
    elif "-h" in (sys.argv):
        bot = DossBot()
        bot.help()
    else:
        fire.Fire(DossBot)
