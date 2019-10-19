from subprocess import Popen, PIPE

if __name__ == '__main__':
    # command_template = "python ipsn_interpolate.py -gl 10 -inter ildw -ildw_dist 0.9 -eo ipsn/error-tmp -gran {} -lo ipsn/inter-{}"
    # granularity = [4]

    command_template = "python ipsn_interpolate.py -gl 40 -inter ildw -ildw_dist 0.9 -eo ipsn/error-tmp -gran {} -lo ipsn/inter-{}"
    granularity = [6, 8, 10, 12, 14, 16, 18]
    ps = []
    for gran in granularity:
        command = command_template.format(gran, gran).split(' ')
        print(' '.join(command))
        p = Popen(command, stdout=PIPE)
        ps.append((p, command))
    for p, command in ps:
        p.wait()
