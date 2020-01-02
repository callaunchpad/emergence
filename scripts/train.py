from symmetric_play.utils.parser import train_parser, args_to_params

parser = train_parser()
args = parser.parse_args()
params = args_to_params(args)

train(params)