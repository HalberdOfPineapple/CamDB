from shutil import copyfile

class ConfigParser(object):

    def __init__(self, cnf):
        f = open(cnf)
        self._cnf = cnf
        self._knobs = {}
        for line in f:
            if line.strip().startswith('skip-external-locking') \
                    or line.strip().startswith('[') \
                    or line.strip().startswith('#') \
                    or line.strip() == '':
                pass
            else:
                try:
                    k, _, v = line.strip().split()
                    self._knobs[k] = v
                except:
                    continue
        f.close()

    def replace(self, tmp='/tmp/tmp.cnf'):
        record_list = []
        f1 = open(self._cnf)
        f2 = open(tmp, 'w')

        for line in f1:
            tpl = line.strip().split()
            if len(tpl) < 1: # empty line: directly copy to target conf
                f2.write(line)
            elif tpl[0] in self._knobs: # knob setting line detected, rewrite its value given the new configuration
                record_list.append(tpl[0])
                tpl[2] = self._knobs[tpl[0]]
                f2.write('%s\t\t%s %s\n' % (tpl[0], tpl[1], tpl[2]))
            else: # other lines, directly copy to target conf
                f2.write(line)

        # Write the configuration that has not been covered by existing settings
        for key in self._knobs.keys():
            if not key in record_list:
                f2.write('%s\t\t%s %s\n' % (key, '=', self._knobs[key]))

        f1.close()
        f2.close()
        copyfile(tmp, self._cnf)

    def set(self, k, v):
        if isinstance(v, str) and ' ' in v:
            self._knobs[k] = "'{}'".format(v)
        else:
            self._knobs[k] = v