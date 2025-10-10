#!/usr/bin/env python
import os, subprocess, threading

METEOR_JAR = 'meteor-1.5.jar'

class Meteor:
    def __init__(self):
        # 1) lock antes de qualquer coisa, para o __del__ não quebrar
        self.lock = threading.Lock()

        # 2) ambiente
        self.env = os.environ.copy()
        self.env['LC_ALL'] = 'en_US.UTF_8'

        # 3) cwd onde fica o JAR
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        jar_path = os.path.join(self.cwd, METEOR_JAR)

        # 4) valida se o JAR existe
        if not os.path.isfile(jar_path):
            raise FileNotFoundError(f"METEOR jar não encontrado: {jar_path}")

        # 5) ORDEM CORRETA: opções da JVM (-Xmx2G) ANTES de -jar
        self.meteor_cmd = [
            'java', '-Xmx2G', '-jar', METEOR_JAR,
            '-', '-', '-stdio', '-l', 'en', '-norm'
        ]

        # 6) inicia o processo
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=self.cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, universal_newlines=True, bufsize=1
        )

    def compute_score(self, gts, res):
        scores = []
        eval_line = 'EVAL'
        self.lock.acquire()
        try:
            for i in range(len(res)):
                assert len(res[i]) == 1
                stat = self._stat(res[i][0], gts[i])
                eval_line += f' ||| {stat}'
            self.meteor_p.stdin.write(eval_line + '\n')
            for _ in range(len(res)):
                scores.append(float(self.meteor_p.stdout.readline().strip()))
            final_score = float(self.meteor_p.stdout.readline().strip())
            return final_score, scores
        finally:
            self.lock.release()

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(score_line + '\n')
        return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        # só tenta fechar se os atributos existirem e o processo estiver vivo
        if hasattr(self, 'lock'):
            try:
                self.lock.acquire()
                if hasattr(self, 'meteor_p') and self.meteor_p and (self.meteor_p.poll() is None):
                    try:
                        self.meteor_p.stdin.close()
                    except Exception:
                        pass
                    self.meteor_p.kill()
                    self.meteor_p.wait(timeout=1)
            except Exception:
                pass
            finally:
                self.lock.release()
