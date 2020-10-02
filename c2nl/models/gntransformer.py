import torch
import torch.nn as nn
import torch.nn.functional as f

from prettytable import PrettyTable
from c2nl.modules.embeddings import Embeddings
from c2nl.models.GNBlock import GAEncoder
from c2nl.models.transformer import TransformerDecoder
from c2nl.inputters import constants
from c2nl.modules.global_attention import GlobalAttention
from c2nl.modules.copy_generator import CopyGenerator, CopyGeneratorCriterion
from c2nl.utils.misc import sequence_mask


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()
        self.enc_input_size = 0
        self.dec_input_size = 0

        self.src_word_embeddings = Embeddings(args.emsize,
                                                args.src_vocab_size+len(constants.NODE_TYPE_VOCAB),
                                                constants.PAD)
        self.enc_input_size += args.emsize

        self.tgt_word_embeddings = Embeddings(args.emsize,
                                                args.tgt_vocab_size,
                                                constants.PAD)
        self.dec_input_size += args.emsize

        self.use_word_type = args.use_word_type
        if self.use_word_type:
            self.word_type_embeddings = nn.Embedding(len(constants.WORD_TYPE_VOCAB),
                                                self.enc_input_size)

        self.src_pos_emb = args.src_pos_emb
        self.tgt_pos_emb = args.tgt_pos_emb
        if self.src_pos_emb:
            self.src_pos_embeddings = nn.Embedding(args.max_src_len+1, 
                                                   self.enc_input_size)
        if self.tgt_pos_emb:
            self.tgt_pos_embeddings = nn.Embedding(args.max_tgt_len + 2,
                                                   self.dec_input_size)

        self.dropout = nn.Dropout(args.dropout_emb)

    def forward(self,
                sequence,
                wtype=None,
                pos_enc=None,
                mode='encoder',
                step=None):

        if mode == 'encoder':
            word_rep = self.src_word_embeddings(sequence.unsqueeze(2))  # B x P x d
            if self.use_word_type:
                word_type_rep = self.word_type_embeddings(wtype)
                word_rep = word_rep + word_type_rep
            if self.src_pos_emb:
                pos_rep = self.src_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep
            word_rep = self.dropout(word_rep)
            word_rep = word_rep.squeeze(0)

        elif mode == 'decoder':
            word_rep = self.tgt_word_embeddings(sequence.unsqueeze(2))  # B x P x d
            if self.tgt_pos_emb:
                if step is None:
                    pos_enc = torch.arange(start=0,
                                           end=word_rep.size(1)).type(torch.LongTensor)
                else:
                    pos_enc = torch.LongTensor([step])  # used in inference time

                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.tgt_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep
                word_rep = self.dropout(word_rep)

        else:
            raise ValueError('Unknown embedder mode!')

        return word_rep


class Encoder(nn.Module):
    def __init__(self,
                 args,
                 input_size):
        super(Encoder, self).__init__()
        try:
            use_rpe=args.use_rpe
            rpe_mode=args.rpe_mode
            rpe_size=args.rpe_size
            rpe_layer=args.rpe_layer
            rpe_share_emb=args.rpe_share_emb
        except:
            use_rpe=False
            rpe_mode=None
            rpe_size=None
            rpe_layer=None
            rpe_share_emb=None
        try: rpe_all=args.rpe_all
        except: rpe_all=True
        self.gat = GAEncoder(num_blocks=args.nlayers,
                             d_model=input_size, 
                             heads=args.num_head,  
                             d_k=args.d_k, 
                             d_v=args.d_v, 
                             d_ff=args.d_ff,
                             num_layers=args.ngatlayers,
                             dropout=args.trans_drop,
                             RPE=use_rpe,
                             RPE_mode=rpe_mode,
                             RPE_size=rpe_size,
                             RPE_layer=rpe_layer,
                             RPE_share_emb=rpe_share_emb,
                             RPE_all=rpe_all)

    def count_parameters(self):
        return self.gat.count_parameters()
    
    def forward(self, bg, h, dm=None, da=None, nids=None, mask=None, seq_length=None):
        return self.gat(bg, h, dm, da, nids, mask, seq_length)


class Decoder(nn.Module):
    def __init__(self, args, input_size):
        super(Decoder, self).__init__()

        self.input_size = input_size

        self.transformer = TransformerDecoder(
            num_layers=args.nlayers,
            d_model=self.input_size,
            heads=args.num_head,
            d_k=args.d_k,
            d_v=args.d_v,
            d_ff=args.d_ff,
            dropout=args.trans_drop
        )

        if args.reload_decoder_state:
            state_dict = torch.load(
                args.reload_decoder_state, map_location=lambda storage, loc: storage
            )
            self.decoder.load_state_dict(state_dict)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def init_decoder(self,
                     src_lens,
                     max_src_len):
        return self.transformer.init_state(src_lens, max_src_len)

    def decode(self,
               tgt_words,
               tgt_emb,
               memory_bank,
               state,
               step=None):

        decoder_outputs, attns = self.transformer(tgt_words,
                                                    tgt_emb,
                                                    memory_bank,
                                                    state,
                                                    step=step)

        return decoder_outputs, attns

    def forward(self,
                memory_bank,
                memory_len,
                tgt_pad_mask,
                tgt_emb):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]
        state = self.init_decoder(memory_len, max_mem_len)
        return self.decode(tgt_pad_mask, tgt_emb, memory_bank, state)


class GNTransformer(nn.Module):
    """Module that writes an answer for the question given a passage."""

    def __init__(self, args, tgt_dict):
        """"Constructor of the class."""
        super(GNTransformer, self).__init__()

        self.name = 'GNTransformer'
        self.embedder = Embedder(args)
        self.encoder = Encoder(args, self.embedder.enc_input_size)
        self.decoder = Decoder(args, self.embedder.dec_input_size)
        self.layer_wise_attn = args.layer_wise_attn

        self.generator = nn.Linear(self.decoder.input_size, args.tgt_vocab_size)
        if args.share_decoder_embeddings:
            assert args.emsize == self.decoder.input_size
            self.generator.weight = self.embedder.tgt_word_embeddings.word_lut.weight

        try:
            self._copy = args.copy_attn
        except:
            self._copy = False
        if self._copy:
            self.copy_attn = GlobalAttention(dim=self.decoder.input_size,
                                             attn_type=args.attn_type)
            self.copy_generator = CopyGenerator(self.decoder.input_size,
                                                tgt_dict,
                                                self.generator)
            self.criterion = CopyGeneratorCriterion(vocab_size=len(tgt_dict),
                                                    force_copy=args.force_copy)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _run_forward_ml(self,
                        code_len,
                        summ_word_rep,
                        summ_len,
                        tgt_seq,
                        wtype,
                        bg, 
                        astok,
                        wpos, 
                        dm,
                        da,
                        nids,
                        mask,
                        seq_length,
                        src_map,
                        alignment,
                        **kwargs):

        batch_size = code_len.size(0)
        # embed and encode the source sequence
        code_rep = self.embedder(astok,
                                 wtype=wtype,
                                 pos_enc=wpos,
                                 mode='encoder')
        memory_bank = self.encoder(bg, code_rep, dm, da, nids, mask, seq_length)  # B x seq_len x h

        # embed and encode the target sequence
        summ_emb = self.embedder(summ_word_rep, mode='decoder')
        summ_pad_mask = ~sequence_mask(summ_len, max_len=summ_emb.size(1))
        enc_outputs = memory_bank
        layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                 code_len,
                                                 summ_pad_mask,
                                                 summ_emb)
        decoder_outputs = layer_wise_dec_out[-1]

        loss = dict()
        target = tgt_seq[:, 1:].contiguous()
        if self._copy:
            # copy_score: batch_size, tgt_len, src_len
            _, copy_score, _ = self.copy_attn(decoder_outputs,
                                              memory_bank,
                                              memory_lengths=code_len,
                                              softmax_weights=False)

            attn_copy = f.softmax(copy_score, dim=-1)
            scores = self.copy_generator(decoder_outputs, attn_copy, src_map)
            scores = scores[:, :-1, :].contiguous()
            ml_loss = self.criterion(scores,
                                     alignment[:, 1:].contiguous(),
                                     target)
        else:
            scores = self.generator(decoder_outputs)  # `batch x tgt_len x vocab_size`
            scores = scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`
            ml_loss = self.criterion(scores.view(-1, scores.size(2)),
                                     target.view(-1))

        ml_loss = ml_loss.view(*scores.size()[:-1])
        ml_loss = ml_loss.mul(target.ne(constants.PAD).float())
        ml_loss = ml_loss.sum(1) * kwargs['example_weights']
        loss['ml_loss'] = ml_loss.mean()
        loss['loss_per_token'] = ml_loss.div((summ_len - 1).float()).mean()

        return loss

    def forward(self,
                code_len,
                summ_word_rep,
                summ_len,
                tgt_seq,
                wtype,
                bg, 
                astok,
                wpos, 
                dm,
                da,
                nids,
                mask,
                seq_length,
                src_map,
                alignment,
                **kwargs):
        """
        Input:
            - code_len: ``(batch_size)``
            - summ_word_rep: ``(batch_size, max_que_len)``
            - summ_len: ``(batch_size)``
            - tgt_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        if self.training:
            return self._run_forward_ml(code_len,
                                        summ_word_rep,
                                        summ_len,
                                        tgt_seq,
                                        wtype,
                                        bg, 
                                        astok,
                                        wpos, 
                                        dm,
                                        da,
                                        nids,
                                        mask,
                                        seq_length,
                                        src_map,
                                        alignment,
                                        **kwargs)

        else:
            return self.decode(code_len,
                               wtype,
                               bg, 
                               astok,
                               wpos, 
                               dm,
                               da,
                               nids,
                               mask,
                               seq_length,
                               src_map,
                               alignment,
                               **kwargs)

    def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words

    def __generate_sequence(self,
                            params,
                            choice='greedy',
                            tgt_words=None):

        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([constants.BOS])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1

        dec_preds = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]
        dec_states = self.decoder.init_decoder(params['src_len'], max_mem_len)

        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn \
            else params['memory_bank']

        # +1 for <EOS> token
        for idx in range(params['max_len'] + 1):
            tgt = self.embedder(tgt_words,
                                mode='decoder',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(constants.PAD)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            step=idx)
            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            if self._copy:
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                  params['memory_bank'],
                                                  memory_lengths=params['src_len'],
                                                  softmax_weights=False)

                attn_copy = f.softmax(copy_score, dim=-1)
                prediction = self.copy_generator(decoder_outputs,
                                                 attn_copy,
                                                 params['src_map'])
                prediction = prediction.squeeze(1)
                for b in range(prediction.size(0)):
                    if params['blank'][b]:
                        blank_b = torch.LongTensor(params['blank'][b])
                        fill_b = torch.LongTensor(params['fill'][b])
                        if use_cuda:
                            blank_b = blank_b.cuda()
                            fill_b = fill_b.cuda()
                        prediction[b].index_add_(0, fill_b,
                                                 prediction[b].index_select(0, blank_b))
                        prediction[b].index_fill_(0, blank_b, 1e-10)

            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = f.softmax(prediction, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))

            words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])

            words = [params['tgt_dict'][w] for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt_words = words.unsqueeze(1)

        return dec_preds, attentions, dec_log_probs

    def decode(self,
               code_len,
               wtype,
               bg, 
               astok,
               wpos, 
               dm,
               da,
               nids,
               mask,
               seq_length,
               src_map,
               alignment,
               **kwargs):

        code_rep = self.embedder(astok,
                                 wtype=wtype,
                                 pos_enc=wpos,
                                 mode='encoder')
        memory_bank = self.encoder(bg, code_rep, dm, da, nids, mask, seq_length)  # B x seq_len x h

        params = dict()
        params['memory_bank'] = memory_bank
        params['layer_wise_outputs'] = None
        params['src_len'] = code_len
        params['source_vocab'] = kwargs['source_vocab']
        params['src_map'] = src_map
        params['src_dict'] = kwargs['src_dict']
        params['tgt_dict'] = kwargs['tgt_dict']
        params['max_len'] = kwargs['max_len']
        params['fill'] = kwargs['fill']
        params['blank'] = kwargs['blank']

        dec_preds, attentions, _ = self.__generate_sequence(params, choice='greedy')
        dec_preds = torch.stack(dec_preds, dim=1)
        # attentions: batch_size x tgt_len x num_heads x src_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_preds,
            'memory_bank': memory_bank,
            'attentions': attentions
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_encoder_parameters(self):
        return self.encoder.count_parameters()

    def count_decoder_parameters(self):
        return self.decoder.count_parameters()

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table
