from collections import namedtuple

import math
import torch
import copy

from fairseq import checkpoint_utils, options, tasks, utils, sequence_scorer


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def modify_sample( sam, expert_idx):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            src_tokens_first = sam['net_input']['src_tokens'][:,:1]
            src_tokens_second = sam['net_input']['src_tokens'][:,1:]
            tmp = torch.LongTensor(sam['net_input']['src_tokens'].shape[0],1).to(device)
            tmp = tmp.fill_(80)
            tmp = tmp.view(sam['net_input']['src_tokens'].shape[0],1)
            sam['net_input']['src_tokens'] = torch.cat([src_tokens_first,tmp, src_tokens_second], dim = 1)

            sam['net_input']['src_lengths'] = sam['net_input']['src_lengths'] + 1
            
            return sam

    
    

def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )

class Generator():
    def __init__(self, data_path, checkpoint_path, user_dir, method):
        self.parser = options.get_generation_parser(interactive=True)
        self.parser.set_defaults(path=checkpoint_path, user_dir=user_dir, beam='1')

        self.args = options.parse_args_and_arch(self.parser, 
            input_args=[data_path]
        )

        utils.import_user_module(self.args)

        if self.args.buffer_size < 1:
            self.args.buffer_size = 1
        if self.args.max_tokens is None and self.args.max_sentences is None:
            self.args.max_sentences = 1

        assert not self.args.sampling or self.args.nbest == self.args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not self.args.max_sentences or self.args.max_sentences <= self.args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        self.use_cuda = torch.cuda.is_available() and not self.args.cpu
        
        self.gen_expert = 3
        
        self.args.method = method

        self.args.task = 'prophetnet_moe'
        
        self.args.nbest = 15
        self.args.beam = 15

        
        self.task = tasks.setup_task(self.args)
        
        


        self.models, self._model_args = checkpoint_utils.load_model_ensemble(
            self.args.path.split(':'),
            arg_overrides=eval(self.args.model_overrides),
            task=self.task,
        )

        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                need_attn=self.args.print_alignment,
            )
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        self.generator = self.task.build_generator(self.args)
        self.scorer = sequence_scorer.SequenceScorer(self.tgt_dict)

        if self.args.remove_bpe == 'gpt2':
            from fairseq.gpt2_bpe.gpt2_encoding import get_encoder
            self.decoder = get_encoder(
                'fairseq/gpt2_bpe/encoder.json',
                'fairseq/gpt2_bpe/vocab.bpe',
            )
            self.encode_fn = lambda x: ' '.join(map(str, self.decoder.encode(x)))
        else:
            self.decoder = None
            self.encode_fn = lambda x: x

        self.align_dict = utils.load_align_dict(self.args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models]
        )
        
        
            
        
    

    def generate(self, inputs, expert):
        start_id = 0
        results = []
        

        for batch in make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            
            
            translations = self.task.inference_step(self.generator, self.models, sample, expert)

            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))
        
        final_result = []
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)

            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                probs = " ".join(map(
                    lambda x: "{:.4f}".format(x),
                    # convert from base e to base 2
                    hypo["positional_scores"]
                    .div_(math.log(2))
                    .tolist(),
                ))
                final_result.append(probs)
        return(final_result)



