#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <ranges>
#include <span>
#include <stack>
#include <type_traits>
#include <utility>
#include <vector>

/*
TODO:
implement rule of 5:
  dtor - done
  cctor
  mctor
  assign
  massign
implement equality and disequality operators for trie_tree
add allocator support
lookup subscript operator
create a generic implementation for larger types, dispatch implementation to
that if needed add noexcept where applicable document time and space
complexities; this implementation is 'optimized' for DENSE prefix trees!
document type constraints, noexcept movability
document that you are allowed to modify an inserted element, but you should keep
it destructable document that iterators are FAT, expensive to copy/create!
migrate adhoc testing to some framework
add some cmake magic
*/

// static std::ostream &errs = std::cerr;

// Helper functions for debugging template metaprograms.
template <class T> auto show();
// By wrapping a deduced constant into this, the compiler will tell us it's
// actual value, not just simply saying that the given static assertion was not
// satisfied.
template <auto Value> struct id { static constexpr auto value = Value; };
template <auto Value> static constexpr auto id_v = id<Value>::value;

// On my platform, the cacheline is 64 bytes. So, I'm sticking to it.
using cacheline = unsigned char[64];

template <class T, class... Elements>
concept one_of = (std::is_same_v<T, Elements> || ...);

// FIXME: We could use this concept to restrict ourselves to the literals that
// can be expressed within the code. In general, this should be more than
// enough. template <class T> concept string_literal_element = one_of<T, const
// char, const wchar_t, const char8_t, const char16_t, const char32_t>;

/// It will tell how many bytes aligned the given offset. I'm also assuming that
/// offset 0 is aligned at 64 bytes.
consteval std::size_t alignofOffset(std::uint64_t startsAt) {
  return std::countr_zero(startsAt) == 64 ? 64 : std::countr_zero(startsAt) + 1;
}
static_assert(alignofOffset(0b000) == 64);
static_assert(alignofOffset(0b001) == 1);
static_assert(alignofOffset(0b010) == 2);
static_assert(alignofOffset(0b011) == 1);
static_assert(alignofOffset(0b100) == 3);

/// It will make the given offset \p x to be aligned to be divisible by \p y.
consteval std::size_t alignUpTo(std::uint64_t x, std::uint64_t y) {
  assert(std::has_single_bit(y));
  for (std::size_t i = 0; i < y; ++i)
    if ((x + i) % y == 0)
      return x + i;

  assert(false && "It should have found it.");
}
static_assert(alignUpTo(0, 1) == 0);
static_assert(alignUpTo(0, 16) == 0);
static_assert(alignUpTo(42, 1) == 42);
static_assert(alignUpTo(42, 2) == 42);
static_assert(alignUpTo(42, 8) == 48);

/// This meta function calculates where a given T should start in the cacheline
/// to meet it's alignment requirements. So, it will introduce the necessary
/// padding before the object. Also calculates a bunch of other properties of
/// the layout.
template <typename T, std::size_t startsAt = 0> struct align_for {
  static constexpr auto StartsAt = startsAt;
  static constexpr auto Sizeof = alignUpTo(startsAt, alignof(T)) - startsAt;
  static constexpr auto Alignof = alignofOffset(startsAt);
  static constexpr auto EndsAt = startsAt + Sizeof;
};

/// This meta function provides a view of a cacheline by managing the lifetime
/// of a wrapped T object at a given position. It will make sure that the given
/// object meetes its size and alignment requirements. It also provides a safe
/// way accessing the object via std::launder.
template <typename T, std::size_t startsAt = 0> struct cacheline_piece {
  using instance_t = T;

  template <typename... Ts>
  static constexpr void construct(cacheline &repr, Ts &&... args) {
    new (&instance(repr)) instance_t(std::forward<Ts>(args)...);
  }
  static constexpr void destruct(cacheline &repr) {
    instance(repr).~instance_t();
  }
  static constexpr instance_t &instance(cacheline &repr) {
    return *(instance_t *)std::launder(repr + startsAt);
  }
  static constexpr const instance_t &instance(const cacheline &repr) {
    return instance(const_cast<cacheline &>(repr));
  }

  static constexpr auto StartsAt = startsAt;
  static constexpr auto Sizeof = sizeof(instance(std::declval<cacheline &>()));
  static constexpr auto Alignof =
      alignof(decltype(instance(std::declval<cacheline &>())));
  static constexpr auto EndsAt = startsAt + Sizeof;
  static_assert(id_v<EndsAt> <= sizeof(cacheline));
  static_assert(startsAt / Alignof * Alignof == startsAt);
};

/// This is a wrapper class, storing the T object directy on the cacheline at
/// the given position.
template <typename T, std::size_t startsAt = 0>
using inplace_item_view = cacheline_piece<T, startsAt>;

/// This is a wrapper class, storing a handle at the cacheline, and the object
/// itself at the heap.
template <typename T, std::size_t startsAt>
class indirect_item_view
    : public cacheline_piece<std::unique_ptr<T>, startsAt> {
  using base = cacheline_piece<std::unique_ptr<T>, startsAt>;

public:
  template <typename... Ts>
  static constexpr void construct(cacheline &repr, Ts &&... args) {
    // errs << "inplace_item_view::construct @ " << &repr << '\n';
    base::construct(repr, std::make_unique<T>(std::forward<Ts>(args)...));
  }
  /*
  static constexpr void destruct(cacheline &repr) {
    // errs << "inplace_item_view::destruct @ " << &repr << '\n';
    base::destruct(repr);
  }
  */
  using base::destruct;
  static constexpr T &instance(cacheline &repr) {
    return *base::instance(repr);
  }
  static constexpr const T &instance(const cacheline &repr) {
    return *base::instance(repr);
  }
};

/// This is a wrapper class, storing the T object at the heap IF it's considered
/// too big to be stored inline in the cacheline. I'm assuming that a pointer is
/// 8 bytes.
template <typename T, std::size_t startsAt = 0>
using item_holder_view =
    std::conditional_t<sizeof(T) <= 8 && alignof(T) <= alignofOffset(startsAt),
                       inplace_item_view<T, startsAt>,
                       indirect_item_view<T, startsAt>>;

/// The metadata of a cacheline. It holds the discriminator and some other stuff
/// deciding if the current cacheline provides storage for a small or big object
/// representation.
struct meta_byte {
  unsigned char is_big : 1;
  unsigned char small_size : 3;
  unsigned char has_value : 1;
};

/// This class provides the functionality for the small version.
/// It consists of an array of 6 pointers to the child nodes, 6 keys, the value
/// (or the handle to the value) and the metadata. The metadata is a single
/// byte, so stored as the last byte of the cacheline for having a more
/// efficient layout for padding bytes. The value and the metadata are at fixed
/// positions! So, they can be safely accessed using either 'views'.
template <typename UPtr, typename Key, typename T> struct small_view {
  static constexpr std::size_t N = 6;
  using ptrs_t = cacheline_piece<std::array<UPtr, N>>;
  using value_t = item_holder_view<T, ptrs_t::EndsAt>;
  using alignment_padding_t = align_for<std::array<Key, N>, value_t::EndsAt>;
  using keys_t =
      cacheline_piece<std::array<Key, N>, alignment_padding_t::EndsAt>;
  static_assert(sizeof(cacheline) - 9 <= keys_t::EndsAt &&
                    keys_t::EndsAt <= sizeof(cacheline) - 2,
                "1-7 padding bytes depending on T");
  using meta_t = cacheline_piece<meta_byte, sizeof(cacheline) - 1>;

  static constexpr decltype(auto) ptrs(cacheline &repr) {
    return ptrs_t::instance(repr);
  }
  static constexpr decltype(auto) ptrs(const cacheline &repr) {
    return ptrs_t::instance(repr);
  }
  static constexpr decltype(auto) keys(cacheline &repr) {
    return keys_t::instance(repr);
  }
  static constexpr decltype(auto) keys(const cacheline &repr) {
    return keys_t::instance(repr);
  }
  static constexpr decltype(auto) value(cacheline &repr) {
    return value_t::instance(repr);
  }
  static constexpr decltype(auto) value(const cacheline &repr) {
    return value_t::instance(repr);
  }
  static constexpr decltype(auto) meta(cacheline &repr) {
    return meta_t::instance(repr);
  }
  static constexpr decltype(auto) meta(const cacheline &repr) {
    return meta_t::instance(repr);
  }

  /// Start the lifetime of each member, except the value.
  /// This function used for transforming from the big representation to the
  /// small representation.
  static constexpr void partial_construct(cacheline &repr) {
    // Don't construct value_t!
    ptrs_t::construct(repr);
    keys_t::construct(repr);
    meta_t::construct(repr); // FIXME: We could probably omit this.
  }
  static constexpr void construct(cacheline &repr) {
    partial_construct(repr);
    value_t::construct(repr);
  }
  static constexpr void partial_destruct(cacheline &repr) {
    // Don't destruct value_t!
    meta_t::destruct(repr);
    keys_t::destruct(repr);
    ptrs_t::destruct(repr);
  }
  static constexpr void destruct(cacheline &repr) {
    partial_destruct(repr);
    value_t::destruct(repr);
  }
};

/// Same as for the small_view.
// Except that this consists of 2 vectors, which are allocating on the heap. The
// value and metadata members are at the very same offset as for the small
// representation.
template <typename UPtr, typename Key, typename T> struct big_view {
  using ptrs_t = cacheline_piece<std::vector<UPtr>>;
  using keys_t = cacheline_piece<std::vector<Key>, ptrs_t::EndsAt>;
  static_assert(keys_t::EndsAt <= 48);
  using value_t = item_holder_view<T, 48>;
  using meta_t = cacheline_piece<meta_byte, sizeof(cacheline) - 1>;

  static constexpr decltype(auto) ptrs(cacheline &repr) {
    return ptrs_t::instance(repr);
  }
  static constexpr decltype(auto) ptrs(const cacheline &repr) {
    return ptrs_t::instance(repr);
  }
  static constexpr decltype(auto) keys(cacheline &repr) {
    return keys_t::instance(repr);
  }
  static constexpr decltype(auto) keys(const cacheline &repr) {
    return keys_t::instance(repr);
  }
  static constexpr decltype(auto) value(cacheline &repr) {
    return value_t::instance(repr);
  }
  static constexpr decltype(auto) value(const cacheline &repr) {
    return value_t::instance(repr);
  }
  static constexpr decltype(auto) meta(cacheline &repr) {
    return meta_t::instance(repr);
  }
  static constexpr decltype(auto) meta(const cacheline &repr) {
    return meta_t::instance(repr);
  }
  static constexpr void partial_construct(cacheline &repr) {
    // Don't construct value_t!
    ptrs_t::construct(repr);
    keys_t::construct(repr);
    meta_t::construct(repr);
  }
  static constexpr void construct(cacheline &repr) {
    partial_construct(repr);
    value_t::construct(repr);
  }
  static constexpr void partial_destruct(cacheline &repr) {
    // Don't destruct value_t!
    meta_t::destruct(repr);
    keys_t::destruct(repr);
    ptrs_t::destruct(repr);
  }
  static constexpr void destruct(cacheline &repr) {
    value_t::destruct(repr);
    partial_destruct(repr);
  }
};

// detail:
namespace detail {
/// This class will test compiletime that for a bunch of instantiations my
/// assumptions are correct. Namely, that the value and metadata are at the same
/// location for any instantiations for the small and big representations.
template <typename T> struct conformance_checker {
  using uptr = std::unique_ptr<unsigned char[]>;
  using small = small_view<uptr, char, T>;
  using big = big_view<uptr, char, T>;
  static_assert(small::value_t::StartsAt == big::value_t::StartsAt);
  static_assert(small::value_t::EndsAt == big::value_t::EndsAt);
  static_assert(small::meta_t::StartsAt == big::meta_t::StartsAt);
  static_assert(small::meta_t::EndsAt == big::meta_t::EndsAt);
};
// For all of these type, the address of meta and value should be the same.
template struct conformance_checker<char>;
template struct conformance_checker<int>;
template struct conformance_checker<long long>;
template struct conformance_checker<std::tuple<long, long, long[44]>>;
template struct conformance_checker<std::tuple<cacheline, cacheline>>;
} // namespace detail

namespace detail {
template <bool IsConst, typename Key, typename T> class trie_iterator;
}

/// This is the actual owning object. This holds a cacheline, aligned
/// appropriately.
template <typename Key, typename T> class trie_node {
  alignas(sizeof(cacheline)) cacheline repr;
  friend class detail::trie_iterator<true, Key, T>;
  friend class detail::trie_iterator<false, Key, T>;
  using trie_node_uptr = std::unique_ptr<trie_node>;

  using small = small_view<trie_node_uptr, Key, T>;
  using big = big_view<trie_node_uptr, Key, T>;

  /// metadata and value will be unconditionally accessed via the small
  /// representation view. We are safe to do so.
  decltype(auto) meta() { return small::meta(repr); }
  decltype(auto) meta() const { return small::meta(repr); }
  decltype(auto) value() { return small::value(repr); }
  decltype(auto) value() const { return small::value(repr); }
  static constexpr unsigned small_capacity = sizeof(small::keys(repr));

  /// I'm exploiting that both std::array and std::vector are contiguous. A
  /// pointer to the data could be used as a 'dummy' iterator.
  Key *keys_begin() {
    if (meta().is_big)
      return big::keys(repr).data();
    return small::keys(repr).data();
  }
  /// I need this, because the small representation uses an array, which might
  /// not be filled completely all the time.
  Key *keys_end() { return keys_begin() + size(); }
  const Key *keys_begin() const {
    return const_cast<trie_node &>(*this).keys_begin();
  }
  const Key *keys_end() const {
    return const_cast<trie_node &>(*this).keys_end();
  }

  /// Doing the same for the pointers.
  trie_node_uptr *ptrs_begin() {
    if (meta().is_big)
      return big::ptrs(repr).data();
    return small::ptrs(repr).data();
  }
  trie_node_uptr *ptrs_end() { return ptrs_begin() + size(); }
  const trie_node_uptr *ptrs_begin() const {
    return const_cast<trie_node &>(*this).ptrs_begin();
  }
  const trie_node_uptr *ptrs_end() const {
    return const_cast<trie_node &>(*this).ptrs_end();
  }

public:
  using key_type = std::basic_string_view<Key>;
  trie_node() {
    // errs << "trie_node ctor @ " << this << '\n';
    small::construct(repr);
  }
  ~trie_node() {
    // errs << "trie_node dtor @ " << this << '\n';
    (meta().is_big) ? big::destruct(repr) : small::destruct(repr);
  }

  std::size_t size() const {
    return meta().is_big ? std::size(big::keys(repr)) : meta().small_size;
  }

  bool empty() const { return !meta().is_big && meta().small_size == 0; }

  /// Calculates the index of a given key. If there is no such key, None
  /// returned.
  static constexpr std::optional<std::size_t> place_of(const trie_node &node,
                                                       Key K) {
    const auto begin = node.keys_begin();
    const auto end = node.keys_end();
    const auto it = std::find(begin, end, K);
    if (it == end)
      return std::nullopt;
    return std::distance(begin, it);
  }

  /// It will return the pointer to the Nth child.
  static constexpr trie_node *ptr_at(trie_node &node, std::size_t Idx) {
    assert(Idx < node.size());
    return (node.ptrs_begin() + Idx)->get();
  }
  static constexpr const trie_node *ptr_at(const trie_node &node,
                                           std::size_t Idx) {
    return ptr_at(const_cast<trie_node &>(node), Idx);
  }

  /// It will traverse the trie, consuming the key sequence step by step. If it
  /// can no longer consume, becouse the key become empty of the next key is not
  /// found it returns what was left of it. Return value: 1) The last node where
  /// it got stuck 2) Not important unless you want efficiently remove dangling
  /// paths. See later. 3) The leftover key sequence. Remarks: If you want to
  /// remove the node pointed by (1), potentially a whole path becomes dangling.
  /// This path starts from the (2).second-th child of the node (2).first. This
  /// is why I refer to it as leaf_root.
  static constexpr std::tuple<trie_node *, std::pair<trie_node *, std::size_t>,
                              key_type>
  traverse(trie_node &node, key_type KeySeq) {
    const auto has_value_or_many_children = [](const trie_node &node) {
      const auto meta = node.meta();
      return meta.has_value || meta.is_big || meta.small_size > 1;
    };

    std::size_t leaf_root_idx = 0;
    trie_node *leaf_root = &node;
    trie_node *current_node = &node;
    for (std::size_t Idx = 0; Idx < std::size(KeySeq); ++Idx) {
      if (auto MaybeIdx = place_of(*current_node, KeySeq[Idx])) {
        if (has_value_or_many_children(*current_node)) {
          leaf_root = current_node;
          leaf_root_idx = MaybeIdx.value();
        }
        current_node = ptr_at(*current_node, MaybeIdx.value());
      } else {
        return {current_node, {leaf_root, leaf_root_idx}, KeySeq.substr(Idx)};
      }
    }

    return {current_node, {leaf_root, leaf_root_idx}, {}};
  }
  static constexpr std::tuple<const trie_node *,
                              std::pair<trie_node *, std::size_t>, key_type>
  traverse(const trie_node &node, key_type KeySeq) {
    return traverse(const_cast<trie_node &>(node), KeySeq);
  }

  /// It will construct the small cachelines for the given key sequence. It
  /// assumes that it should start building new nodes from the given node and it
  /// has to construct at least one.
  static constexpr trie_node *traverse_create(trie_node &node,
                                              key_type KeySeq) {
    assert(!std::empty(KeySeq));
    assert(!place_of(node, KeySeq.front()) && "should be new");
    trie_node *current_node = &node;

    // FIXME: I sohuld hoist the complex transormation logic out of the loop, as
    // that could only occure only at the **first** node. All the rest will be
    // small.
    for (std::size_t Idx = 0; Idx < std::size(KeySeq); ++Idx) {
      std::unique_ptr new_node = std::make_unique<trie_node>();
      // errs << "created trie_node @ " << new_node.get();
      // errs << " for " << KeySeq[Idx] << '\n';
      trie_node *prev_node = current_node;
      current_node = new_node.get();
      if (prev_node->meta().is_big) {
        // Just append.
        // errs << "just appending to the already big repr\n";
        big::keys(prev_node->repr).push_back(KeySeq[Idx]);
        big::ptrs(prev_node->repr).push_back(std::move(new_node));
      } else {
        unsigned size = prev_node->meta().small_size;
        if (size < small_capacity) {
          // Just append.
          // errs << "just appending to the already small repr\n";
          small::keys(prev_node->repr)[size] = KeySeq[Idx];
          small::ptrs(prev_node->repr)[size] = std::move(new_node);
          ++prev_node->meta().small_size;
        } else {
          // small -> big
          // errs << "transforming small -> big\n";
          const meta_byte new_meta = [=]() {
            auto tmp = prev_node->meta();
            tmp.is_big = true;
            return tmp;
          }();
          // Unfortunately, we have to make temporal copy here, since we will
          // overwrite the cacheline.
          typename big::ptrs_t::instance_t dst_ptrs;
          typename big::keys_t::instance_t dst_keys;

          // FIXME: The big representation should reserve place for some
          // cachelines. I'm not doing it right now.
          std::move(prev_node->ptrs_begin(), prev_node->ptrs_end(),
                    std::back_inserter(dst_ptrs));
          std::move(prev_node->keys_begin(), prev_node->keys_end(),
                    std::back_inserter(dst_keys));

          // value_t is kept in place - by using the partial construct/destruct.
          small::partial_destruct(prev_node->repr);
          big::partial_construct(prev_node->repr);

          big::ptrs(prev_node->repr) = std::move(dst_ptrs);
          big::keys(prev_node->repr) = std::move(dst_keys);

          prev_node->meta() =
              new_meta; // FIXME: I could omit this, by changing partial_*.
          big::keys(prev_node->repr).push_back(KeySeq[Idx]);
          big::ptrs(prev_node->repr).push_back(std::move(new_node));
        }
      }
    }
    return current_node;
  }

  bool contains(key_type KeySeq) const {
    // FIXME: I would use '_', but that's actually forbidden by the standard. I
    // would use std::ignore, but that won't work with structured bindings. -.-
    auto [node, unused, restKeys] = traverse(*this, KeySeq);
    return std::empty(restKeys) ? node->meta().has_value : false;
  }

  template <typename... Ts> bool emplace(key_type KeySeq, Ts &&... args) {
    auto [node, unused, restKeys] = traverse(*this, KeySeq);
    if (!std::empty(restKeys))
      node = traverse_create(*node, restKeys);

    // If we already have value associated with it, bail out.
    if (node->meta().has_value)
      return false;

    // Yey, construct the value in place! (It starts it's lifetime.)
    new (&node->value()) T(std::forward<Ts>(args)...);
    // errs << "constructed T within trie_node @ " << node;
    // errs << " with value " << node->value() << '\n';
    assert(!node->meta().has_value);
    node->meta().has_value = true;
    return true;
  }

  bool remove(key_type KeySeq) {
    auto [leaf, leaf_root_info, restKeys] = traverse(*this, KeySeq);
    if (!std::empty(restKeys))
      return false;

    const auto leaf_meta = leaf->meta();

    // If there are nodes below this, just destroy the associated value and keep
    // the node.
    if (!leaf->empty()) {
      if (leaf_meta.has_value) {
        // errs << "destroyed T within trie_node @ " << leaf;
        // errs << " with value " << leaf->value() << '\n';
        small::value_t::destruct(leaf->repr);
        leaf->meta().has_value = false;
        return true;
      }
    } else {
      assert(!leaf_meta.is_big && "a leaf node must be small");
      assert(leaf_meta.has_value && "the leaf must have a value");
      // Destroy all the unneeded parents.
      auto [leaf_root, idx] = leaf_root_info;
      // errs << "destroying all trie_node starting from @ ";
      // errs << leaf_root << " no. " << idx << " child\n";
      leaf_root->ptrs_begin()[idx].reset();

      // Move everything back by one.
      // errs << "moving everything back by one\n";
      std::move(leaf_root->ptrs_begin() + idx + 1, leaf_root->ptrs_end(),
                leaf_root->ptrs_begin() + idx);
      std::move(leaf_root->keys_begin() + idx + 1, leaf_root->keys_end(),
                leaf_root->keys_begin() + idx);

      // Actually remove the element.
      if (leaf_root->meta().is_big) {
        // errs << "leaf_root had big representation, so I'm popping once\n";
        big::ptrs(leaf_root->repr).pop_back();
        big::keys(leaf_root->repr).pop_back();

        // Transform big -> small if small become enough.
        if (leaf_root->size() <= small_capacity) {
          // errs << "transforming big -> small\n";
          // Move to a temporary location.
          // errs << "moving out the entire (big) ptrs and keys vectors\n";
          typename big::ptrs_t::instance_t src_ptrs =
              std::move(big::ptrs(leaf_root->repr));
          typename big::keys_t::instance_t src_keys =
              std::move(big::keys(leaf_root->repr));
          // Save and set up the metadata.
          meta_byte new_meta = leaf_root->meta();
          new_meta.is_big = false;
          new_meta.small_size = src_keys.size();

          // Do the transformation.
          // errs << "big::partial_destruct\n";
          big::partial_destruct(leaf_root->repr);
          // errs << "small::partial_construct\n";
          small::partial_construct(leaf_root->repr);

          // Move the data back to the small representation.
          // errs << "moving the data back to the small repr\n";
          std::move(std::begin(src_ptrs), std::end(src_ptrs),
                    leaf_root->ptrs_begin());
          std::move(std::begin(src_keys), std::end(src_keys),
                    leaf_root->keys_begin());
          leaf_root->meta() = new_meta;
        } else {
          // errs << "it was still too big to transform to small representation\n";
        }
      } else {
        leaf_root->meta().small_size--;
      }
      return true;
    }

    return false;
  }
};

/// A trie tree is like a trie node, but with some caveats.
/// We will keep track the length of the longest key ever stored into this tree.
/// This will give a good upperbound of the maximal height of the tree. Also,
/// tracking the number of values successfully inserted could also help
/// sometimes.
template <typename Key, typename T>
class trie_tree : private trie_node<Key, T> {
  static_assert(requires { sizeof(Key) == 1; });
  static_assert(requires { std::is_trivially_copyable_v<Key>; });
  static_assert(requires { std::is_trivially_constructible_v<Key>; });
  std::size_t max_height = 0;
  std::size_t value_count = 0;
  using Self = trie_node<Key, T>;
  friend class detail::trie_iterator<true, Key, T>;
  friend class detail::trie_iterator<false, Key, T>;

public:
  trie_tree() = default;

  // I'm not using std::span<const Key> here, because it would not work well
  // with string literals - especially with their null terminator character.
  // Should we consider the null character as an identifier or not. Using
  // string_view makes this interface **much** cleaner.
  using key_type = typename Self::key_type;
  using mapped_type = T;

  constexpr std::size_t size() const noexcept { return value_count; }
  constexpr bool empty() const noexcept { return value_count == 0; }

  /// Prettymuch what the node version does, plus some bookkeeping.
  bool remove(key_type KeySeq) {
    const bool res = Self::remove(KeySeq);

    if (res) {
      assert(value_count > 0);
      --value_count;
    }
    return res;
  }

  /// Contains is just the same as the one for the node version.
  using Self::contains;

  std::size_t count(key_type KeySeq) const { return contains(KeySeq); }

  template <typename... Ts> bool emplace(key_type KeySeq, Ts &&... args) {
    max_height = std::max(max_height, KeySeq.size());
    const bool res = Self::emplace(KeySeq, std::forward<Ts>(args)...);
    if (res)
      ++value_count;
    return res;
  }

  /// Our belowed **fat** iterators.
  using iterator = detail::trie_iterator<false, Key, T>;
  using const_iterator = detail::trie_iterator<true, Key, T>;
  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;
};

namespace detail {
template <bool IsConst, typename Key, typename T> class trie_iterator {
public:
  using value_ptr = std::conditional_t<IsConst, const T *, T *>;

  /// We can use basic_string for aggregating the keys along a path since we
  /// know that the keys are restricted to a single byte trivial* type. So, it's
  /// notionally correct to assume that they will need the API of a 'string'. It
  /// would be odd in the general case though.
  using value_type = std::pair<std::basic_string<Key>, value_ptr>;

  /// I'm implementing only the forward iteration. It could be bidirectional
  /// though.
  using iterator_category = std::forward_iterator_tag;

private:
  using node_t = trie_node<Key, T>;
  /// Durring the DFS walk, we store at each level which child we took last, and
  /// at which node we are.
  using state_t = std::vector<std::pair<node_t *, std::size_t>>;
  state_t state;

  /// From the top-most state, walks down to the left-most node (aka. to the
  /// left-most leaf).
  void descend() {
    assert(!state.empty());
    auto [current, unused] = state.back();
    // errs << "descending from @ " << current << " to @ ";
    // As long as we are not a leaf, continue.
    while (!current->empty()) {
      // Choose the first child, which is at index 0.
      current = current->ptrs_begin()->get();
      state.push_back({current, 0});
    }
    // We either have a leaf which has a value OR we are the ROOT.
    assert(current->meta().has_value || state.size() == 1);
    // errs << current << '\n';
  }

public:
  explicit trie_iterator() = default;
  trie_iterator(trie_iterator &&) = default;
  // STL likes to copy iterators instead of moving. However, copying this
  // iterator will trigger an allocation unfortunately.
  trie_iterator(const trie_iterator &other) {
    // We are copying manually! The default generated would reserve size()
    // amount - which would not guarantee that the longest path would still fit
    // without reallocation if we walk further doen the tree. So, I'm copying
    // here manually.
    state.reserve(other.state.capacity());
    std::copy(other.state.begin(), other.state.end(),
              std::back_inserter(state));
  }
  explicit trie_iterator(trie_tree<Key, T> &tree) {
    // If the tree is empty, return the end iterator.
    if (tree.empty() && !tree.meta().has_value)
      return;

    state.reserve(tree.max_height);
    state.push_back({&tree, 0});
    descend(); // Looks for the first node that holds a value. (The left-most
               // leaf)
    assert(state.back().first->meta().has_value);
  }

  // It would be nice to lazily compute the aggregated key sequence for the
  // given node - only if requested. For example if she only want to print the
  // values, then aggregating the key sequence was just a waist of time PLUS an
  // allocation! Which is wasteful, but whatever.
  value_type operator*() const {
    assert(!state.empty());

    value_type res;
    res.first.reserve(state.size() - 1);
    res.second = &state.back().first->value();

    for (std::size_t i = 1; i < state.size(); ++i) {
      node_t *parent = state[i - 1].first;
      Key current_key = parent->keys_begin()[state[i].second];
      res.first.push_back(current_key);
    }
    return res;
  }

  // FIXME: Implement operator->().

  // It would be too expensive to implement, due to the fact that copying this
  // iterator triggers an allocation!
  trie_iterator operator++(int) = delete;
  trie_iterator operator--(int) = delete;

  // Move to the next node that has an associated value.
  trie_iterator &operator++() {
    // errs << "operator++\n";
    node_t *current;
    // Search, until we either find one with a value OR we hit the ROOT node.
    do {
      auto [prev_visited_child, visited_child_idx] = state.back();
      state.pop_back();
      std::tie(current, std::ignore) = state.back();
      // errs << "current " << current << ", prev " << prev_visited_child;
      // errs << " (child no. " << visited_child_idx << ")\n";
      // errs << "current->size(): " << current->size() << '\n';
      const auto new_child_idx = visited_child_idx + 1;
      // Try the next child.
      if (new_child_idx < current->size()) {
        // errs << "choosing child no. " << new_child_idx << '\n';
        auto unvisited_child_it = current->ptrs_begin() + new_child_idx;
        assert(unvisited_child_it->get() && "child must exist");
        state.push_back({unvisited_child_it->get(), new_child_idx});
        descend();
        return *this;
      }

      // Move further up...
      // errs << "moving to the parent...\n";
    } while (!current->meta().has_value && state.size() > 1);

    // If we have root only, without holding a value, just return the end
    // iterator, aka. empty state.
    if (state.size() == 1 && !state.back().first->meta().has_value) {
      state.pop_back();
      assert(state.empty());
    }

    return *this;
  }

  // FIXME: Implement this for bidirectional iterators.
  trie_iterator &operator--() = delete;

  // The defaulted would be fine, but inefficient. In DFS walk if the top-most
  // state matches we are golden.
  bool operator==(const trie_iterator &other) const {
    if (!state.empty() && !other.state.empty())
      return state.back() == other.state.back();
    // Falling back to the default equality is cheap now, one of them must have
    // zero states.
    return state == other.state;
  }
  bool operator!=(const trie_iterator &other) const {
    return !(*this == other);
  }
};
} // namespace detail

// It would be awesome to have this closer to the class, but we have to break
// the circular dependancy of the declarations.
template <typename Key, typename T>
typename trie_tree<Key, T>::iterator trie_tree<Key, T>::begin() {
  return trie_tree<Key, T>::iterator{*this};
}
template <typename Key, typename T>
typename trie_tree<Key, T>::const_iterator trie_tree<Key, T>::begin() const {
  return trie_tree<Key, T>::const_iterator{*this};
}
template <typename Key, typename T>
typename trie_tree<Key, T>::iterator trie_tree<Key, T>::end() {
  return trie_tree<Key, T>::iterator{};
}
template <typename Key, typename T>
typename trie_tree<Key, T>::const_iterator trie_tree<Key, T>::end() const {
  return trie_tree<Key, T>::const_iterator{};
}

int main() {
  bool tmp;
  trie_tree<char, int> node;
  tmp = node.contains("wow");
  assert(!tmp);
  tmp = node.emplace("wow", 42);
  assert(tmp);
  tmp = node.contains("wow");
  assert(tmp);
  tmp = node.contains("wo");
  assert(!tmp);
  tmp = node.emplace("wo", 43);
  assert(tmp);
  tmp = node.remove("w");
  assert(!tmp);

  for (char ch = 'A'; ch <= 'G'; ++ch) {
    char str[] = {'w', 'o', ch};
    tmp = node.emplace(std::string_view{str, std::size(str)}, (int)ch);
    assert(tmp);
  }

  {
    // FIXME: Use copy constructor and/or copy assignment when implemented.
    trie_tree<char, int> copy;
    assert(!node.empty());
    for (auto &&[KeySeq, Value] : node) {
      copy.emplace(KeySeq, *Value);
    }
    assert(*((*node.begin()).second) == *((*copy.begin()).second));
    *((*copy.begin()).second) = 404;
    assert(*((*node.begin()).second) != *((*copy.begin()).second));
    // FIXME: Use equlity operator when implemented.
    assert(copy.size() == node.size());
  }

  for (char ch = 'C'; ch <= 'Z'; ++ch) {
    std::cout << "removing " << ch << "\n";
    char str[] = {'w', 'o', ch};
    tmp = node.remove(std::string_view{str, std::size(str)});
    assert(ch <= 'G' ? tmp : !tmp);
  }

  tmp = node.size() == 4 && node.contains("wo") && node.contains("wow") &&
        node.contains("woA") && node.contains("woB");
  assert(tmp);

  assert(node.begin() != node.end());
  assert(node.begin() == node.begin());
  assert(++node.begin() != node.begin());
  assert(++node.begin() == ++node.begin());
  auto it = node.begin();
  auto it2 = it;
  assert(it == it2);

  std::cout << "=== after inserting wow and wo ===\n";
  for (auto &&[KeySeq, Value] : node) {
    for (auto Key : KeySeq)
      std::cout << Key;
    std::cout << " --> " << *Value << '\n';
  }

  tmp = node.remove("woof");
  assert(!tmp);
  tmp = node.remove("wo");
  assert(tmp);

  tmp = node.emplace("woof", 40);
  tmp = node.emplace("woaf", 45);
  tmp = node.emplace("wobf", 44);
  tmp = node.emplace("wobf", 48);

  std::cout << "=== after removing woof and wo ===\n";
  for (auto &&[KeySeq, Value] : node) {
    for (auto Key : KeySeq)
      std::cout << Key;
    std::cout << " --> " << *Value << '\n';
  }

  trie_tree<char, int> xxx;
  assert(xxx.begin() == xxx.end());
}
